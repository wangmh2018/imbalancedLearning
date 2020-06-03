package Xmeans;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.IOException;

public class KmeansRun {
    public static void run(String centerPath, String dataPath, String newCenterPath, boolean runReduce) throws IOException, ClassNotFoundException, InterruptedException {
        Configuration conf = new Configuration();
        conf.set("centersPath", centerPath);

        Job job = new Job(conf, "myKmeans");
        job.setJarByClass(KmeansRun.class);

        job.setMapperClass(KmeansMapper.class);

        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(Text.class);

        if (runReduce) {
            job.setReducerClass(KmeansReducer.class);
            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(Text.class);
        }

        FileInputFormat.addInputPath(job, new Path(dataPath));
        FileOutputFormat.setOutputPath(job, new Path(newCenterPath));
        System.out.println(job.waitForCompletion(true));
    }

    public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
        String centerPath = "hdfs://10.187.86.242:9000/wmh/KmeansTestCenters.csv";
        String dataPath = "hdfs://10.187.86.242:9000/wmh/KmeansTest.csv";
        String newCenterPath = "hdfs://10.187.86.242:9000/wmh/kmeansTestOutput";

        int count = 0;

        while(true){
            run(centerPath,dataPath,newCenterPath,true);
            System.out.println("第"+ ++count +"次计算");
            if (Utils.compareCenters(centerPath,newCenterPath)){
                run(centerPath,dataPath,newCenterPath,false);
                break;
            }

        }
    }
}
