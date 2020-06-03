import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import scala.util.Random


object unequilibrium {


  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)

    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)

    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.WARN)

    //    自定义类型
    type instance = (List[Double], String)

    //    计算熵值
    def calcEntropy(a: Array[Double]): Double = {
      var result = 0.0
      for (i <- 0 until a.length) {
        var temp = a(i) * (math.log(a(i)) / math.log(2))
        result += temp
      }
      -result
    }

    //    隶属度除法函数
    def memShipDevide(a: Array[(Array[Double], Double, Array[Double])], kNearest: Int): ArrayBuffer[Double] = {
      //    k为近邻数
      val k = kNearest
      //    m为隶属度维度
      val m = 2
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

    //    训练样例和类中心的距离计算函数
    def trainAndCenterDis(trainInstance: Array[Double], Center: Array[Double]): Double = {
      var arr = new Array[Double](trainInstance.length - 1)
      for (i <- 0 until trainInstance.length - 1) {
        arr(i) = trainInstance(i)
      }
      math.sqrt((Center, arr).zipped.map((ri, si) => math.pow((ri - si), 2)).reduce(_ + _))

    }

    //    类中心计算函数
    def labelCenterDevide(a: Array[Double]): Array[Double] = {
      val result = new Array[Double](a.length - 1)
      val num = a(a.length - 1)
      for (i <- 0 until a.length - 1) {
        result(i) = a(i) / num
      }
      result
    }

    //    def entropyThreshold(i: Double, n: Double): Double = {
    //      val result = 1 - (i / n * math.exp(i * n) / 10)
    //      result
    //    }

    //    距离计算函数
    def Distance(a: Array[Double], b: Array[Double]): Double = {
      if (a.length != b.length) {
        throw new Exception("Dimensions do not match")
      }
      else {
        math.sqrt((a, b).zipped.map((ri, si) => math.pow((ri - si), 2)).reduce(_ + _))
      }
    }

    //selectInstance为使用CFKNN算法进行样例选择后的每个partition的训练集，positiveInstance为正类样例集合，upsampleNum为s，kNearest为k个近邻,m为在计算danger数据集时用到的近邻数
    //    upsampleNum必须小于等于KNearest
    def borderlineSmoteTwo(selectInstance: ArrayBuffer[(String, List[Double])], positiveInstance: Array[(String, List[Double])], upsampleNum: Int, kNearest: Int, M: Int): ArrayBuffer[(String, List[Double])] = {
      //      newInstance用来存储新生成的样例
      val newInstance = ArrayBuffer[(String, List[Double])]()

      //      保存danger数据集
      val dangerInstance = ArrayBuffer[(String, List[Double])]()
      //对每个正类样例在训练集中找到M个近邻
      for (i_instance <- positiveInstance) {
        var instanceDis = ArrayBuffer[((String, List[Double]), (String, List[Double]), Double)]()
        for (j_instance <- selectInstance) {
          if (i_instance != j_instance) {
            val dis = Distance(i_instance._2.toArray, j_instance._2.toArray)
            val temp = (i_instance, j_instance, dis)
            instanceDis += temp
          }
        }
        instanceDis = instanceDis.sortBy(_._3)
        //此处M应该设为较大的数，即对每个正类样例在训练集中找到前M个近邻，M设置的较小会导致没有样例选入danger样例集
        instanceDis.trimEnd(instanceDis.length - M)
        var negativeInstanceNum = 0
        //计算M个近邻中属于负类样例的个数
        for (i_num <- 0 until M) {
          if (instanceDis(i_num)._2._1 == "2.00000") {
            negativeInstanceNum += 1
          }
        }
        //此处在原borderlineSmote算法上做了修改，原算法为M/2，但由于靠近边界样例较少，所以改为此，可以将样例归为danger样例
        if (negativeInstanceNum >= (M / 6).toInt) {
          dangerInstance += i_instance
        }
      }

      //      对每个danger样例集中的数据进行样例生成
      for (danger <- dangerInstance) {
        val randomNum = randomNew(upsampleNum, kNearest)
        var dangerInstanceDis = ArrayBuffer[((String, List[Double]), (String, List[Double]), Double)]()
        for (positive <- positiveInstance) {
          val dis = Distance(danger._2.toArray, positive._2.toArray)
          val temp = (danger, positive, dis)
          dangerInstanceDis += temp
        }
        dangerInstanceDis = dangerInstanceDis.sortBy(_._3)
        dangerInstanceDis.trimEnd(dangerInstanceDis.length - kNearest)
        for (index <- randomNum) {
          if (dangerInstance.contains(dangerInstanceDis(index)._2)) {
            val zeroToOneNum = Random.nextDouble()
            val sampleInstance = ArrayBuffer[Double]()
            for (i <- 0 until danger._2.length) {
              val temp = (danger._2(i) - dangerInstanceDis(index)._2._2(i)) * zeroToOneNum + danger._2(i)
              sampleInstance += temp
            }
            val instance = (danger._1, sampleInstance.toList)
            newInstance += instance

          }
          else {
            val zeroToPointFiveNum = Random.nextDouble() / 2
            val sampleInstance = ArrayBuffer[Double]()
            for (i <- 0 until danger._2.length) {
              val temp = (danger._2(i) - dangerInstanceDis(index)._2._2(i)) * zeroToPointFiveNum + danger._2(i)
              sampleInstance += temp
            }
            val instance = (danger._1, sampleInstance.toList)
            newInstance += instance

          }
        }
      }
      //      println(newInstance.length)
      newInstance
    }

    //    用KNN算法，并返回测试精度
    def knnAccuracy(trainSet: ArrayBuffer[(String, List[Double])], testSet: ArrayBuffer[(String, List[Double])], kNearest: Int): Double = {
      var accuracy: Double = 0.0
      val testSetNum = testSet.length
      var trueNum = 0
      for (test <- testSet) {
        var testAndTrainInstanceDis = ArrayBuffer[((String, List[Double]), (String, List[Double]), Double)]()
        for (train <- trainSet) {
          val dis = Distance(test._2.toArray, train._2.toArray)
          val temp = (test, train, dis)
          testAndTrainInstanceDis += temp
        }
        testAndTrainInstanceDis = testAndTrainInstanceDis.sortBy(_._3)
        testAndTrainInstanceDis.trimEnd(testAndTrainInstanceDis.length - kNearest)
        var labelOneNum = 0
        var labelTwoNum = 0
        for (instance <- testAndTrainInstanceDis) {
          if (instance._2._1 == "1.00000") {
            labelOneNum += 1
          }
          else {
            labelTwoNum += 1
          }
        }
        if (labelOneNum >= labelTwoNum && test._1 == "1.00000") {
          trueNum += 1
        }
        if (labelOneNum < labelTwoNum && test._1 == "2.00000") {
          trueNum += 1
        }
      }
      accuracy = trueNum / testSetNum
      accuracy

    }

    //    使用knn算法标记当前测试样例
    def knnAlgorithm(testInstance: (String, List[Double]), trainSet: ArrayBuffer[(String, List[Double])], kNearest: Int): instance = {
      var testAndTrainDis = ArrayBuffer[((String, List[Double]), (String, List[Double]), Double)]()
      for (train <- trainSet) {
        val dis = Distance(testInstance._2.toArray, train._2.toArray)
        val temp = (testInstance, train, dis)
        testAndTrainDis += temp
      }
      testAndTrainDis = testAndTrainDis.sortBy(_._3)
      testAndTrainDis.trimEnd(testAndTrainDis.length - kNearest)
      var labelOneNum = 0
      var labelTwoNum = 0
      for (tuple <- testAndTrainDis) {
        if (tuple._2._1 == "1.00000") {
          labelOneNum += 1
        }
        else {
          labelTwoNum += 1
        }
      }

      if (labelOneNum >= labelTwoNum) {
        val result: instance = (testInstance._2, "1.00000")
        result
      }
      else {
        val result: instance = (testInstance._2, "2.00000")
        result
      }

    }

    //    数组求和
    def addArray(a: Array[Double], b: Array[Double]): Array[Double] = {
      val sum = new Array[Double](a.length)
      for (k <- 0 until a.length) {
        sum(k) = a(k) + b(k)
      }
      sum
    }

    def randomNew(n: Int, kNearest: Int) = {
      var temp = kNearest
      var resultList: List[Int] = Nil
      while (resultList.length < n) {
        val randomNum = (new Random).nextInt(temp - 1)
        resultList = resultList ::: List(randomNum)
        temp -= 1
      }
      resultList
    }


    def trainInsMemberShipCalc(a: List[((String, List[Double]), Double)], kNearNum: Int): ArrayBuffer[(Array[Double], Double, Array[Double])] = {
      //      设置近邻数为3

      //      此处针对二分类问题，需要新建两个变长数组，分别存储两类中心，对于多分类问题，此处需要修改
      var labelOneCenter = new ArrayBuffer[Double]()
      var labelTwoCenter = new ArrayBuffer[Double]()
      //            a(0)._1._2为样例属性
      for (i <- 0 until a(0)._1._2.length) {
        var labelOneSum = 0.0
        var labelTwoSum = 0.0
        for (j <- 0 until kNearNum) {
          //          判断类标值，对于类标不同的数据集，此处需要修改
          if (a(j)._1._1 == "1.00000") {
            labelOneSum += a(j)._1._2(i)
          }
          else if (a(j)._1._1 == "2.00000") {
            labelTwoSum += a(j)._1._2(i)
          }
        }
        labelOneCenter += labelOneSum
        labelTwoCenter += labelTwoSum
      }

      //      求出每一类的中心
      for (i <- 0 until labelOneCenter.length) {
        if (labelOneCenter(labelOneCenter.length - 1) != 0) {
          labelOneCenter(i) = labelOneCenter(i) / labelOneCenter(labelOneCenter.length - 1)
        }
        if (labelTwoCenter(labelTwoCenter.length - 1) != 0) {
          labelTwoCenter(i) = labelTwoCenter(i) / labelTwoCenter(labelTwoCenter.length - 1)
        }
      }

      val trainInsMemberShip = new ArrayBuffer[(Array[Double], Double, Array[Double])]()
      for (i <- 0 until kNearNum) {
        var insAndLabelOneCenterDis = Distance(a(i)._1._2.toArray, labelOneCenter.toArray)
        var insAndLabelTwoCenterDis = Distance(a(i)._1._2.toArray, labelTwoCenter.toArray)
        if (insAndLabelOneCenterDis == 0) {
          insAndLabelOneCenterDis = Double.MaxValue
        }
        else {
          insAndLabelOneCenterDis = math.pow(insAndLabelOneCenterDis, -2)
        }
        if (insAndLabelTwoCenterDis == 0) {
          insAndLabelTwoCenterDis = Double.MaxValue
        }
        else {
          insAndLabelTwoCenterDis = math.pow(insAndLabelTwoCenterDis, -2)
        }
        val sum = insAndLabelOneCenterDis + insAndLabelTwoCenterDis
        val instance_i_membership = new ArrayBuffer[Double]()
        instance_i_membership += insAndLabelOneCenterDis / sum
        instance_i_membership += insAndLabelTwoCenterDis / sum
        //        println(instance_i_membership)
        val temp = (a(i)._1._2.toArray, a(i)._2, instance_i_membership.toArray)
        trainInsMemberShip += temp
      }
      trainInsMemberShip
    }

    val conf = new SparkConf().setAppName("unequilibrium").setMaster("local")

    conf.set("spark.defalut.parallelism", "3")


    val sc = new SparkContext(conf)

    type myType = (((String, List[Double]), List[Double]), Double)
    val kNearest = 3
    val alapha = 0.8

    val testSet = Source.fromFile("F:\\test.txt")
    val testInstance = ArrayBuffer[(String, List[Double])]()
    for (line <- testSet.getLines) {

      val arr = line.split(" ")


      val attribute = new Array[Double](arr.length)
      for (i <- 0 until arr.length - 1) {
        attribute(i) = arr(i + 1).toDouble
      }

      //  属性值加一列便于统计样例个数，缩小RDD的内存开销
      attribute(arr.length - 1) = 1

      val label = arr(0).toString
      //println(label)
      val temp = (label, attribute.toList)
      //      println(temp)
      testInstance += temp
      //      println(testInstance)
    }
    val testInstanceBroadcast = sc.broadcast(testInstance)


    val trainInitRDD = sc.textFile("F:\\train.txt", 3).map(line => {
      val arr = line.split(" ")
      val attribute = new Array[Double](arr.length)
      for (i <- 0 until arr.length - 1) {
        attribute(i) = arr(i + 1).toDouble
      }
      //  属性值加一列便于统计样例个数，缩小RDD的内存开销
      attribute(arr.length - 1) = 1
      val label = arr(0).toString
      (label, attribute)
    })

    //    负类样例类标为2.00000
    var negativeSampleInsRDD = trainInitRDD.combineByKey((v: Array[Double]) => List(v),
      (c: List[Array[Double]], v: Array[Double]) => v :: c,
      (c1: List[Array[Double]], c2: List[Array[Double]]) => c1 ::: c2).map(line => {

      val temp = new ArrayBuffer[(String, List[Double])]()
      //      每类初始化的样例个数
      if (line._1 == "2.00000") {
        for (i <- 0 until line._2.length) {
          val a = (line._1, line._2(i).toList)
          temp += a

        }
      }
      temp
    }).flatMap(line => line)

    //    selectOneInsRDD.foreach(line => println(line))

    var positiveSampleInsRDD = trainInitRDD.map(line => {
      (line._1, line._2.toList)
    }).subtract(negativeSampleInsRDD)

    val positiveSampleInsBroadcast = sc.broadcast(positiveSampleInsRDD.collect())

    val result = negativeSampleInsRDD.coalesce(numPartitions = 3, shuffle = true).mapPartitionsWithIndex((num, line) => {

      val testInstanceResult = ArrayBuffer[instance]()
      var trainInstance = ArrayBuffer[(String, List[Double])]()
      val negativeInstance = ArrayBuffer[(String, List[Double])]()
      val positiveInstance = ArrayBuffer[(String, List[Double])]()
      //      用来保存对应类别的样例个数

      while (line.hasNext) {

        negativeInstance += line.next()
      }

      for (i <- 0 until positiveSampleInsBroadcast.value.length) {

        positiveInstance += positiveSampleInsBroadcast.value(i)
      }

      if (negativeInstance.length / positiveInstance.length >= 2) {

        val k = positiveInstance.length
        val tempList = randomNew(k, 20)

        for (i <- 0 until k) {

          trainInstance += negativeInstance(tempList(i))
          negativeInstance -= negativeInstance(tempList(i))
        }

        trainInstance ++= positiveInstance

        for (element <- negativeInstance) {
          var insDistance = ArrayBuffer[(((String, List[Double]), Double))]()
          for (j <- 0 until trainInstance.length) {
            val dis = Distance(trainInstance(j)._2.toArray, element._2.toArray)
            val temp = (trainInstance(j), dis)
            insDistance += temp
          }
          insDistance = insDistance.sortBy(_._2)
          insDistance.trimEnd(insDistance.length - kNearest)

          val memshipResult = memShipDevide(trainInsMemberShipCalc(insDistance.toList, kNearNum = kNearest).toArray, kNearest)

          val entropy = calcEntropy(memshipResult.toArray)

          if (entropy > 0.9) {
            trainInstance += element
          }

        }

        val upsampleInstance = borderlineSmoteTwo(trainInstance, positiveSampleInsBroadcast.value, 5, 6, 10)

        trainInstance ++= upsampleInstance

        for (testinstance <- testInstanceBroadcast.value) {
          val result = knnAlgorithm(testinstance, trainInstance, 3)
          testInstanceResult += result
        }
      }
      testInstanceResult.toIterator
    }).combineByKey((v: String) => List(v),
      (c: List[String], v: String) => v :: c,
      (c1: List[String], c2: List[String]) => c1 ::: c2).map(line => {
      var labelOneNum = 0
      var labelTwoNum = 0
      for (label <- line._2) {
        if (label == "1.00000") {
          labelOneNum += 1
        }
        else {
          labelTwoNum += 1
        }
      }
      if (labelOneNum > labelTwoNum) {
        ("1.00000", line._1)
      }
      else {
        ("2.00000", line._1)
      }
    }).filter(line => {
      testInstanceBroadcast.value.contains(line)
    }).count
    val accuracy = result.toDouble / testInstanceBroadcast.value.length.toDouble

    println(accuracy)
  }
}
