import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object XmeansAndFknn {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)

    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)

    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.WARN)


    /*
    instanceSet:样例集合
    center：对应簇中心
    n：训练集包含样例个数
    K：簇的个数
    return:噪声方差
     */
    def noiseVariance(instanceSet: List[List[Double]], center: Array[Double], n: Int, K: Int): Double = {
      //      distance_sum保存样例到中心的距离和
      var distance_sum = 0.0
      for (instance <- instanceSet) {
        distance_sum += Distance(instance.toArray, center)
      }
      val result = distance_sum / (n - K)
      result
    }

    /*

     */
    def logLikelihood(k: Int, n: Int, n_cluster_num: Int, d: Double, variance: Double): Double = {
      val p1 = -n_cluster_num * Math.log(2 * Math.PI)
      val p2 = -n_cluster_num * d * Math.log(variance)
      val p3 = -(n_cluster_num - k)
      val p4 = n_cluster_num * Math.log(n_cluster_num)
      val p5 = -n_cluster_num * Math.log(n)
      val loglike = (p1 + p2 + p3) / 2 + p4 + p5
      loglike
    }

    def bic(k: Int, n: Int, d: Double, variance: Double, clusterSize: Array[Int]): Double = {
      var L = 0.0
      for (i <- 0 until k) {
        L += logLikelihood(k, n, clusterSize(i), d, variance)
      }
      L
    }

    def addArray(a: Array[Double], b: Array[Double]): Array[Double] = {
      val sum = new Array[Double](a.length)
      for (k <- 0 until a.length) {
        sum(k) = a(k) + b(k)
      }
      sum
    }

    def centerCalc(cluster: List[List[Double]]): Array[Double] = {
      var result = new Array[Double](cluster(0).length)
      val instanceNum = cluster.length
      for (instance <- cluster) {
        result = addArray(result, instance.toArray)
      }
      for (i <- 0 until result.length) {
        result(i) = result(i) / instanceNum
      }
      result
    }

    def Distance(a: Array[Double], b: Array[Double]): Double = {
      if (a.length != b.length) {
        throw new Exception("Dimensions do not match")
      }
      else {
        math.sqrt((a, b).zipped.map((ri, si) => math.pow((ri - si), 2)).reduce(_ + _))
      }
    }

    def judgeSplit(parentRdd: RDD[List[Double]], sampleNum: Int, K:Int,d: Double, parentParamNum: Double, subParamNum: Double, numCluster: Int, numIterations: Int, label: Int, filePath: String, fileNameVar: Int): List[RDD[List[Double]]] = {
      val parentCombineRdd = parentRdd.map(line => (1, line)).combineByKey((v: List[Double]) => List(v),
        (c: List[List[Double]], v: List[Double]) => v :: c,
        (c1: List[List[Double]], c2: List[List[Double]]) => c1 ::: c2)

      val clusterSize = parentCombineRdd.map(line => line._2.length).collect()

      val parentCenter = parentCombineRdd.map(line => {
        val center = centerCalc(line._2)

        (line, center)
      })

      val parentVariance = parentCenter.map(line => {
        val variance = noiseVariance(line._1._2, line._2, sampleNum, K)
        variance
      }).collect()

      val parentBic = bic(1, sampleNum, d, parentVariance(0), clusterSize) - 0.5 * parentParamNum * Math.log(sampleNum)

      val parentVector = parentRdd.map(line => Vectors.dense(line.toArray)).cache()

      val parentKMeansModel = KMeans.train(parentVector, numCluster, numIterations)


      val parentKmeansResultCombine = parentVector.map(line => {
        val prediction = parentKMeansModel.predict(line)
        (prediction, line.toArray.toList)
      }).combineByKey((v: List[Double]) => List(v),
        (c: List[List[Double]], v: List[Double]) => v :: c,
        (c1: List[List[Double]], c2: List[List[Double]]) => c1 ::: c2)


      val parentKmeansResultClusterNum = parentKmeansResultCombine.map(line => line._2.length).collect()

      val parentKmeansResultVariance = parentKmeansResultCombine.map(line => {
        val center = centerCalc(line._2)
        (line, center)
      }).map(line => {
        val variance = noiseVariance(line._1._2, line._2, sampleNum, K)
        variance
      }).reduce(_ + _)


      val parentKmeansBicScore = bic(parentKmeansResultClusterNum.length, sampleNum, d, parentKmeansResultVariance, parentKmeansResultClusterNum) - 0.5 * subParamNum * Math.log(sampleNum)

      if (parentKmeansBicScore > parentBic) {

        val labelZeroCluster = parentVector.map(line => {
          val prediction = parentKMeansModel.predict(line)
          (prediction, line.toArray.toList)
        }).filter(_._1 == 0).map(_._2)

        val labelOneCluster = parentVector.map(line => {
          val prediction = parentKMeansModel.predict(line)
          (prediction, line.toArray.toList)
        }).filter(_._1 == 1).map(_._2)


        val result = List[RDD[List[Double]]](labelZeroCluster, labelOneCluster)

        return result
      }
      else {
        parentVector.map(line => {
          val prediction = parentKMeansModel.predict(line) + label
          val temp = line.toArray.toBuffer
          temp ++= ArrayBuffer(prediction)
          val result = line.toArray.toList :+ prediction
          result
        }).map(line => line.mkString(",")).saveAsTextFile(filePath + fileNameVar.toString)
        val result = List[RDD[List[Double]]]()
        return result
      }
    }

    //以上是Xmeans所用函数       以上是Xmeans所用函数       以上是Xmeans所用函数       以上是Xmeans所用函数
    def calcEntropy(a: Array[Double]): Double = {
      var result = 0.0
      for (i <- 0 until a.length) {
        var temp = a(i) * (math.log(a(i)) / math.log(2))
        result += temp
      }
      -result
    }

    def memShipDevide(a: Array[(Array[Double], Double, Array[Double])], kNearest: Int, classNum: Int): ArrayBuffer[Double] = {
      //    k为近邻数
      val k = kNearest
      //    m为隶属度维度
      val m = classNum
      //    构造分子变长数组
      val numeratorarr = ArrayBuffer[Double]()
      //    构造分母变长数组
      val denominatorsumarr = ArrayBuffer[Double]()
      //    构造隶属度变长数组
      val resultarr = ArrayBuffer[Double]()
      //    初始化分子变长数组
      for (j <- 0 until m) {
        var sum = 0.0
        for (i <- 0 until k) {
          //    会导致nan，加1
          val result = 1 / (a(i)._2 + 0.000001) * a(i)._3(j)
          sum += result
        }
        numeratorarr += sum
      }
      //   初始化分母变长数组
      for (j <- 0 until m) {
        var sum = 0.0
        for (i <- 0 until k) {
          //   会导致nan，加1
          sum += 1 / (a(i)._2 + 0.000001)
        }
        denominatorsumarr += sum
      }
      //  初始化隶属度数组
      for (i <- 0 until m) {
        var result = numeratorarr(i) / denominatorsumarr(i)
        resultarr += result
      }
      resultarr
    }

    def randomNew(n: Int) = {
      var resultList: List[Int] = Nil
      while (resultList.length < n) {
        val randomNum = (new Random).nextInt(20)
        if (!resultList.exists(s => s == randomNum)) {
          resultList = resultList ::: List(randomNum)
        }
      }
      resultList
    }

    def trainInsMemberShipCalc(a: List[((String, List[Double]), Double)], kNearNum: Int): ArrayBuffer[(Array[Double], Double, Array[Double])] = {
      //      设置近邻数为3

      //      此处针对二分类问题，需要新建两个变长数组，分别存储两类中心，对于多分类问题，此处需要修改

      var label0Center = new ArrayBuffer[Double]()
      var label1Center = new ArrayBuffer[Double]()
      var label2Center = new ArrayBuffer[Double]()
      var label3Center = new ArrayBuffer[Double]()
      var label4Center = new ArrayBuffer[Double]()
      var label5Center = new ArrayBuffer[Double]()
      var label6Center = new ArrayBuffer[Double]()
      var label7Center = new ArrayBuffer[Double]()
      var label8Center = new ArrayBuffer[Double]()
      var label9Center = new ArrayBuffer[Double]()

      for (i <- 0 until a(0)._1._2.length) {

        var label0Sum = 0.0
        var label1Sum = 0.0
        var label2Sum = 0.0
        var label3Sum = 0.0
        var label4Sum = 0.0
        var label5Sum = 0.0
        var label6Sum = 0.0
        var label7Sum = 0.0
        var label8Sum = 0.0
        var label9Sum = 0.0

        for (j <- 0 until kNearNum) {
          //          判断类标值，对于类标不同的数据集，此处需要修改

          if (a(j)._1._1 == "0") {
            label0Sum += a(j)._1._2(i)
          }
          else if (a(j)._1._1 == "1") {
            label1Sum += a(j)._1._2(i)
          }
          else if (a(j)._1._1 == "2") {
            label2Sum += a(j)._1._2(i)
          }
          else if (a(j)._1._1 == "3") {
            label3Sum += a(j)._1._2(i)
          }
          else if (a(j)._1._1 == "4") {
            label4Sum += a(j)._1._2(i)
          }
          else if (a(j)._1._1 == "5") {
            label5Sum += a(j)._1._2(i)
          }
          else if (a(j)._1._1 == "6") {
            label6Sum += a(j)._1._2(i)
          }
          else if (a(j)._1._1 == "7") {
            label7Sum += a(j)._1._2(i)
          }
          else if (a(j)._1._1 == "8") {
            label8Sum += a(j)._1._2(i)
          }
          else if (a(j)._1._1 == "9") {
            label9Sum += a(j)._1._2(i)
          }

        }
        label0Center += label0Sum
        label1Center += label1Sum
        label2Center += label2Sum
        label3Center += label3Sum
        label4Center += label4Sum
        label5Center += label5Sum
        label6Center += label6Sum
        label7Center += label7Sum
        label8Center += label8Sum
        label9Center += label9Sum
      }
      //      求出每一类的中心
      for (i <- 0 until label1Center.length) {

        if (label0Center(label0Center.length - 1) != 0) {
          label0Center(i) = label0Center(i) / label0Center(label0Center.length - 1)
        }
        if (label1Center(label1Center.length - 1) != 0) {
          label1Center(i) = label1Center(i) / label1Center(label1Center.length - 1)
        }
        if (label2Center(label2Center.length - 1) != 0) {
          label2Center(i) = label2Center(i) / label2Center(label2Center.length - 1)
        }
        if (label3Center(label3Center.length - 1) != 0) {
          label3Center(i) = label3Center(i) / label3Center(label3Center.length - 1)
        }
        if (label4Center(label4Center.length - 1) != 0) {
          label4Center(i) = label4Center(i) / label4Center(label4Center.length - 1)
        }
        if (label5Center(label5Center.length - 1) != 0) {
          label5Center(i) = label5Center(i) / label5Center(label5Center.length - 1)
        }
        if (label6Center(label6Center.length - 1) != 0) {
          label6Center(i) = label6Center(i) / label6Center(label6Center.length - 1)
        }
        if (label7Center(label7Center.length - 1) != 0) {
          label7Center(i) = label7Center(i) / label7Center(label7Center.length - 1)
        }
        if (label8Center(label8Center.length - 1) != 0) {
          label8Center(i) = label8Center(i) / label8Center(label8Center.length - 1)
        }
        if (label9Center(label9Center.length - 1) != 0) {
          label9Center(i) = label9Center(i) / label9Center(label9Center.length - 1)
        }

      }
      val trainInsMemberShip = new ArrayBuffer[(Array[Double], Double, Array[Double])]()
      for (i <- 0 until kNearNum) {
        var insAndLabel0CenterDis = Distance(a(i)._1._2.toArray, label0Center.toArray)
        var insAndLabel1CenterDis = Distance(a(i)._1._2.toArray, label1Center.toArray)
        var insAndLabel2CenterDis = Distance(a(i)._1._2.toArray, label2Center.toArray)
        var insAndLabel3CenterDis = Distance(a(i)._1._2.toArray, label3Center.toArray)
        var insAndLabel4CenterDis = Distance(a(i)._1._2.toArray, label4Center.toArray)
        var insAndLabel5CenterDis = Distance(a(i)._1._2.toArray, label5Center.toArray)
        var insAndLabel6CenterDis = Distance(a(i)._1._2.toArray, label6Center.toArray)
        var insAndLabel7CenterDis = Distance(a(i)._1._2.toArray, label7Center.toArray)
        var insAndLabel8CenterDis = Distance(a(i)._1._2.toArray, label8Center.toArray)
        var insAndLabel9CenterDis = Distance(a(i)._1._2.toArray, label9Center.toArray)

        if (insAndLabel0CenterDis == 0) {
          insAndLabel0CenterDis = Double.MaxValue
        }
        else {
          insAndLabel0CenterDis = math.pow(insAndLabel0CenterDis, -2)
        }
        if (insAndLabel1CenterDis == 0) {
          insAndLabel1CenterDis = Double.MaxValue
        }
        else {
          insAndLabel1CenterDis = math.pow(insAndLabel1CenterDis, -2)
        }
        if (insAndLabel2CenterDis == 0) {
          insAndLabel2CenterDis = Double.MaxValue
        }
        else {
          insAndLabel2CenterDis = math.pow(insAndLabel2CenterDis, -2)
        }
        if (insAndLabel3CenterDis == 0) {
          insAndLabel3CenterDis = Double.MaxValue
        }
        else {
          insAndLabel3CenterDis = math.pow(insAndLabel3CenterDis, -2)
        }
        if (insAndLabel4CenterDis == 0) {
          insAndLabel4CenterDis = Double.MaxValue
        }
        else {
          insAndLabel4CenterDis = math.pow(insAndLabel4CenterDis, -2)
        }
        if (insAndLabel5CenterDis == 0) {
          insAndLabel5CenterDis = Double.MaxValue
        }
        else {
          insAndLabel5CenterDis = math.pow(insAndLabel5CenterDis, -2)
        }
        if (insAndLabel6CenterDis == 0) {
          insAndLabel6CenterDis = Double.MaxValue
        }
        else {
          insAndLabel6CenterDis = math.pow(insAndLabel6CenterDis, -2)
        }
        if (insAndLabel7CenterDis == 0) {
          insAndLabel7CenterDis = Double.MaxValue
        }
        else {
          insAndLabel7CenterDis = math.pow(insAndLabel7CenterDis, -2)
        }
        if (insAndLabel8CenterDis == 0) {
          insAndLabel8CenterDis = Double.MaxValue
        }
        else {
          insAndLabel8CenterDis = math.pow(insAndLabel8CenterDis, -2)
        }
        if (insAndLabel9CenterDis == 0) {
          insAndLabel9CenterDis = Double.MaxValue
        }
        else {
          insAndLabel9CenterDis = math.pow(insAndLabel9CenterDis, -2)
        }

        val sum = insAndLabel0CenterDis + insAndLabel1CenterDis + insAndLabel2CenterDis + insAndLabel3CenterDis + insAndLabel4CenterDis + insAndLabel5CenterDis + insAndLabel6CenterDis + insAndLabel7CenterDis + insAndLabel8CenterDis + insAndLabel9CenterDis
        val instance_i_membership = new ArrayBuffer[Double]()

        instance_i_membership += insAndLabel0CenterDis / sum
        instance_i_membership += insAndLabel1CenterDis / sum
        instance_i_membership += insAndLabel2CenterDis / sum
        instance_i_membership += insAndLabel3CenterDis / sum
        instance_i_membership += insAndLabel4CenterDis / sum
        instance_i_membership += insAndLabel5CenterDis / sum
        instance_i_membership += insAndLabel6CenterDis / sum
        instance_i_membership += insAndLabel7CenterDis / sum
        instance_i_membership += insAndLabel8CenterDis / sum
        instance_i_membership += insAndLabel9CenterDis / sum

        val temp = (a(i)._1._2.toArray, a(i)._2, instance_i_membership.toArray)
        trainInsMemberShip += temp
      }
      trainInsMemberShip
    }

    val conf = new SparkConf().setAppName("Xmeans")
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set("spark.shuffle.consolidateFiles", "true")
    conf.set("spark.memory.useLegacyMode", "true")
    conf.set("spark.shuffle.file.buffer", "128kb")
    conf.set("spark.shuffle.memoryFraction", "0.1")
    conf.set("spark.storage.memoryFraction", "0.4")
    conf.set("spark.shuffle.consolidateFiles", "leqiu true")
    conf.set("spark.shuffle.sort.bypassMergeThreshold", "200")
    //    conf.set("spark.storage.safetyFaction", "0.9")
    //    conf.set("spark.shuffle.safteyFraction", "0.9")
    conf.registerKryoClasses(Array(classOf[(((String, List[Double]), Double))], classOf[String], classOf[List[Double]], classOf[Array[Double]], classOf[ArrayBuffer[Double]], classOf[(String, List[Double])], classOf[(String, Array[Double])]))
    conf.set("spark.reducer.maxSizeInFlight", "100m")

    val sc = new SparkContext(conf)


    var trainSet: RDD[(String, Array[Double])] = null
    if (args(1) == "F" && args(2) == " ") {
      println("类标第一列，空格分隔")
      val temp = sc.textFile(args(0), minPartitions = 196).map(line => {
        val arr = line.split(" ")
        val attribute = new Array[Double](arr.length)
        for (i <- 0 until arr.length - 1) {
          attribute(i) = arr(i + 1).toDouble
        }
        attribute(arr.length - 1) = 1
        val label = arr(0).toString
        (label, attribute)
      })
      trainSet = temp
    }


    else if (args(1) == "F" && args(2) == ",") {
      println("类标第一列，逗号分隔")
      val temp = sc.textFile(args(0), minPartitions = 196).map(line => {
        val arr = line.split(",")
        val attribute = new Array[Double](arr.length)
        for (i <- 0 until arr.length - 1) {
          attribute(i) = arr(i + 1).toDouble
        }
        attribute(arr.length - 1) = 1
        val label = arr(0).toString
        (label, attribute)
      })
      trainSet = temp
    }

    else if (args(1) == "L" && args(2) == " ") {
      println("类标最后一列，空格分隔")
      val temp = sc.textFile(args(0), minPartitions = 196).map(line => {
        val arr = line.split(" ")
        val attribute = new Array[Double](arr.length)
        for (i <- 0 until arr.length - 1) {
          attribute(i) = arr(i).toDouble
        }
        attribute(arr.length - 1) = 1
        val label = arr(arr.length - 1).toString
        (label, attribute)
      })
      trainSet = temp
    }

    else if (args(1) == "L" && args(2) == ",") {
      println("类标最后一列，逗号分隔")
      val temp = sc.textFile(args(0), minPartitions = 196).map(line => {
        val arr = line.split(",")
        val attribute = new Array[Double](arr.length)
        for (i <- 0 until arr.length - 1) {
          attribute(i) = arr(i).toDouble
        }
        attribute(arr.length - 1) = 1
        val label = arr(arr.length - 1).toString
        (label, attribute)
      })
      trainSet = temp
    }

    val d = args(3).toDouble
    val parentParam = 1.0 + 1.0 * d
    val subParam = 2.0 + 2.0 * d
    val negativeSampleNum = trainSet.count().toInt

    val data1 = trainSet.map(line => Vectors.dense(line._2)).persist(StorageLevel.MEMORY_ONLY_SER)

    //    设置Kmeans参数，训练Kmeans模型
    val numCluster = 2
    val numIterations = args(4).toInt
    val initKmeansModel = KMeans.train(data1, numCluster, numIterations)

    val labelOneIns = data1.map(line => {
      val prediction = initKmeansModel.predict(line)
      (prediction,line.toArray.toList)
    }).filter(_._1==1).map(_._2)

    val labelZeroIns = data1.map(line => {
      val prediction = initKmeansModel.predict(line)
      (prediction,line.toArray.toList)
    }).filter(_._1==0).map(_._2)

    var rddList = List[RDD[List[Double]]](labelZeroIns, labelOneIns)
    var fileNameTemp = 1
    var labelTemp = 0
    val xmeansFilePath = args(5)

    while (!rddList.isEmpty && labelTemp != 10) {
      val parentRdd = rddList.head
      val temp = judgeSplit(parentRdd, negativeSampleNum, rddList.length,d, parentParam, subParam, numCluster, numIterations, labelTemp, xmeansFilePath, fileNameTemp)
      rddList = rddList.drop(1)
      if (!temp.isEmpty) {
        rddList = rddList ::: temp
      }

      else {
        fileNameTemp += 1
        labelTemp += 2
      }

    }
    if (labelTemp == 10 && !rddList.isEmpty) {
      var result: RDD[String]= null
      for (i <- 0 until rddList.length) {
        if (i==0){
          result=rddList(i).map(line => {
            val temp = line :+ 9
            temp.mkString(",")
          })
        }
        else {
          val a = rddList(i).map(line => {
            val temp = line :+ 9
            temp.mkString(",")
          })
          result.union(a)
        }

      }
      result.saveAsTextFile(xmeansFilePath+fileNameTemp.toString)
    }

    val kNearest = args(6).toInt

    val trainInitRDD = sc.textFile(args(7), minPartitions = 196).map(line => {
      val arr = line.split(",")
      val attribute = new Array[Double](arr.length - 1)
      for (i <- 0 until arr.length - 1) {
        attribute(i) = arr(i).toDouble
      }
      val label = arr(arr.length - 1).toString

      (label, attribute)
    }).persist(StorageLevel.MEMORY_ONLY_SER)

    var selectInsRDD = trainInitRDD.combineByKey((v: Array[Double]) => List(v),
      (c: List[Array[Double]], v: Array[Double]) => v :: c,
      (c1: List[Array[Double]], c2: List[Array[Double]]) => c1 ::: c2, numPartitions = 196).map(line => {
      //      每类初始化的样例个数
      val k = args(8).toInt
      val temp = new ArrayBuffer[(String, List[Double])]()
      val tempList = randomNew(k)
      for (i <- 0 until k) {
        val a = (line._1, line._2(tempList(i)).toList)
        temp += a
      }
      temp
    }).flatMap(line => line).persist(StorageLevel.MEMORY_ONLY_SER)

    var tRDD = trainInitRDD.map(line => {
      (line._1, line._2.toList)
    }).subtract(selectInsRDD, numPartitions = 196).persist(StorageLevel.MEMORY_ONLY_SER)

    var alapha = 0.0
    if (labelTemp == 2) {
      alapha = 0.9
    }
    if (labelTemp == 4) {
      alapha = 1.6
    }
    if (labelTemp == 6) {
      alapha = 1.7
    }
    if (labelTemp == 8) {
      alapha = 1.9
    }
    if (labelTemp == 10) {
      alapha = 2.1
    }
    for (i_num <- 1 to 5) {
      val selectInsbroad = sc.broadcast(selectInsRDD.collect())

      val tEntropyAndSelectRDD = tRDD.map(line => {

        var temp = new ArrayBuffer[(((String, List[Double]), Double))]()
        for (i <- 0 until selectInsbroad.value.length) {
          val dis = Distance(selectInsbroad.value(i)._2.toArray, line._2.toArray)
          val a = ((selectInsbroad.value(i), dis))
          temp += a
        }
        temp = temp.sortBy(_._2)
        temp.trimEnd(temp.length - kNearest)
        (line, temp)
      }).map(line => {

        val temp = (line._1, memShipDevide(trainInsMemberShipCalc(line._2.toList, kNearNum = kNearest).toArray, kNearest, labelTemp))

        val entropy = calcEntropy(temp._2.toArray)

        val testInsSelectToTrainIns = new ArrayBuffer[(String, List[Double])]()
        if (entropy > alapha) {
          testInsSelectToTrainIns += line._1
        }
        testInsSelectToTrainIns
      }).flatMap(line => line).persist(StorageLevel.MEMORY_ONLY_SER)

      selectInsRDD = selectInsRDD.union(tEntropyAndSelectRDD).persist(StorageLevel.MEMORY_ONLY_SER)
      tRDD = tRDD.subtract(tEntropyAndSelectRDD, numPartitions = 196).persist(StorageLevel.MEMORY_ONLY_SER)
      selectInsbroad.unpersist

      val resultRDD = selectInsRDD.map(line => {
        val result = ArrayBuffer[Double]()
        result ++= line._2.toBuffer
        result += line._1.toDouble
        result
      }).map(line => line.mkString(",")).repartition(196).saveAsTextFile(args(9) + alapha + i_num.toString)

      if (labelTemp == 2) {
        alapha = alapha - 0.04
      }
      if (labelTemp == 4) {
        alapha = alapha - 0.105
      }
      if (labelTemp == 6) {
        alapha = alapha - 0.12
      }
      if (labelTemp == 8) {
        alapha = alapha - 0.2
      }
      if (labelTemp == 10) {
        alapha = alapha - 0.22
      }
    }


  }
}
