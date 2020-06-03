import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.clustering._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils._
import org.apache.spark.mllib.util.{KMeansDataGenerator, MLUtils}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}
import spark.DataFrameTest.initKmeansModel

import scala.collection.mutable._
import scala.tools.nsc.Variance

object Xmeans {
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

    def judgeSplit(parentRdd: RDD[List[Double]], sampleNum: Int, d: Double, parentParamNum: Double, subParamNum: Double, numCluster: Int, numIterations: Int, label: Int, filePath: String, fileNameVar: Int): List[RDD[List[Double]]] = {
      val parentCombineRdd = parentRdd.map(line => (1, line)).combineByKey((v: List[Double]) => List(v),
        (c: List[List[Double]], v: List[Double]) => v :: c,
        (c1: List[List[Double]], c2: List[List[Double]]) => c1 ::: c2)

      val clusterSize = parentCombineRdd.map(line => line._2.length).collect()

      val parentCenter = parentCombineRdd.map(line => {
        val center = centerCalc(line._2)

        (line, center)
      })

      val parentVariance = parentCenter.map(line => {
        val variance = noiseVariance(line._1._2, line._2, sampleNum, 1)
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
        val variance = noiseVariance(line._1._2, line._2, sampleNum, 2)
        variance
      }).reduce(_ + _)


      val parentKmeansBicScore = bic(2, sampleNum, d, parentKmeansResultVariance, parentKmeansResultClusterNum) - 0.5 * subParamNum * Math.log(sampleNum)

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

    val conf = new SparkConf().setAppName("Xmeans").setMaster("local")
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set("spark.shuffle.consolidateFiles", "true")
    conf.set("spark.memory.useLegacyMode", "true")
    conf.set("spark.shuffle.file.buffer", "128kb")
    conf.set("spark.shuffle.memoryFraction", "0.2")
    conf.set("spark.storage.memoryFraction", "0.08")
    conf.set("spark.shuffle.consolidateFiles", "leqiu true")
    conf.set("spark.shuffle.sort.bypassMergeThreshold", "200")
    conf.set("spark.storage.safetyFaction", "0.9")
    conf.set("spark.shuffle.safteyFraction", "0.9")
    conf.registerKryoClasses(Array(classOf[String], classOf[List[Double]], classOf[Array[Double]], classOf[ArrayBuffer[Double]]))
    conf.set("spark.reducer.maxSizeInFlight", "100m")

    val sc = new SparkContext(conf)

    val trainSet = sc.textFile("/Users/b/Desktop/标准化数据集/MiniBooNE_PID/negative.csv", minPartitions = 140).map(line => {
      val arr = line.split(",")
      val attribute = new Array[Double](arr.length)
      for (i <- 0 until arr.length - 1) {
        attribute(i) = arr(i).toDouble
      }
      attribute(arr.length - 1) = 1
      val label = arr(arr.length - 1).toString
      (label, attribute)
    })

    val d = 50.0
    val parentParam = 1.0 + 1.0 * d
    val subParam = 2.0 + 2.0 * d
    val negativeSampleNum = trainSet.count().toInt

    val data1 = trainSet.map(line => Vectors.dense(line._2)).persist(StorageLevel.MEMORY_ONLY_SER)

    //    设置Kmeans参数，训练Kmeans模型
    val numCluster = 2
    val numIterations = 30
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
    val xmeansFilePath = "/Users/b/Documents/xmeans_test/result_"

    while (!rddList.isEmpty && labelTemp != 10) {
      val parentRdd = rddList.head
      val temp = judgeSplit(parentRdd, negativeSampleNum, d, parentParam, subParam, numCluster, numIterations, labelTemp, xmeansFilePath, fileNameTemp)
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


  }
}
