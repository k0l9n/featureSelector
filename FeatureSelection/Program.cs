using Accord.Genetic;
using Accord.Math;
using Accord.Neuro;
using Accord.Neuro.Learning;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using Accord.Statistics.Filters;

namespace FeatureSelection
{
    abstract class InformativeFitnessFuncion : IFitnessFunction
    {
        protected double[][] TrainInputs;
        protected double[][] TrainOutpus;

        protected double[][] ValidationInputs;
        protected double[][] ValidationOutpus;

        protected double[][] OnlyFeaturesTrainInputs; //rename
        protected double[][] OnlyFeaturesValidInputs;

        protected double TrainPart { get; set; } = 0.66;

        public void TakeInfornativeFeatures(IChromosome chromosome)
        {
            var cromo = (BinaryChromosome)chromosome;

            var tempLearning = new List<double[]>();
            var tempTesting = new List<double[]>();

            for (int j = 0; j < cromo.Length; j++)
            {

                if ((cromo.Value & (ulong)Math.Pow(2, j)) != 0)
                {
                    tempLearning.Add(TrainInputs.Select(x => x[j]).ToArray());
                    tempTesting.Add(ValidationInputs.Select(x => x[j]).ToArray());
                }
            }
            OnlyFeaturesTrainInputs = tempLearning.ToArray().Transpose();
            OnlyFeaturesValidInputs = tempTesting.ToArray().Transpose();
        }
        public void TrainValidationSplitData(double[][] features, double[][] results)
        {
            var rowsCount = features.Rows();

            var takeForLearning = (int)Math.Truncate((rowsCount - 1) * TrainPart);
            var takeForTest = rowsCount - (takeForLearning + 1);

            TrainInputs = features.ToList().Take(takeForLearning).ToArray();
            TrainOutpus = results.ToList().Take(takeForLearning).ToArray();

            features.ToList().RemoveRange(0, takeForLearning);
            results.ToList().RemoveRange(0, takeForLearning);
            ValidationInputs = features.Take(takeForTest).ToArray();
            ValidationOutpus = results.Take(takeForTest).ToArray();           
        }

        public void NormalizeData()
        {
            Normalization normalization = new Normalization();
            normalization.Detect(TrainInputs);
            normalization.ApplyInPlace(TrainInputs);
            normalization.ApplyInPlace(ValidationInputs);
            
        }
        public abstract double Evaluate(IChromosome chromosome);
    }

    class EvaluationResult
    {
        public EvaluationResult()
        {
            ErrorHistory = new List<double>();
        }
        public int CurrentIterationCount { get; set; }
        public double CurrentError { get; set; }
        public List<double> ErrorHistory { get; set; }
        public bool IsErrorOnTest { get; set; }

    }


    class InformativeFitness : InformativeFitnessFuncion
    {
        public double Momentum { get; set; } = 0;
        public double LearningRate { get; set; } = 0.1;

        public int MaxEpochs { get; set; } = 15;

        public int[] NetworkLayers { get; set; }

        public bool AutoShuffle { get; set; }

        #region EvaluationStopFactor

        public int MaxIterationCount { get; set; }

        private double _errorPart;

        public double ErrorPart
        {
            get => _errorPart;
            set
            {
                if (value < 0)
                {
                    _errorPart = 0;
                }
                else if(value > 100)
                {
                    _errorPart = 100;
                }
                else
                {
                    _errorPart = value;
                }
            }
        }

        public int ErrorPartNotChanged { get; set; }

        #endregion

        public InformativeFitness(List<string> data)
        {
            //var data = FileWorker.ReadFile(@"C:\Users\Nick\Desktop\data\data\optdigits.txt");
            data.Shuffle();
            var args = new List<double[]>();
            var results = new List<double[]>();
            string tmp;
            foreach (var str in data)
            {
                tmp = str.Substring(str.Length - 1);
                var a = new[] { -0.6, -0.6, -0.6, -0.6, -0.6, -0.6, -0.6, -0.6, -0.6, -0.6 }; //10
                var counter = Int32.Parse(tmp);
                a[counter] = 0.6;
                results.Add(a);
                args.Add(str.Substring(0, str.Length - 2).Split(',')
                    .Select(x => (Double.Parse(x, CultureInfo.InvariantCulture) - 8) / 8).ToArray());
            }
            
            TrainInputs = args.Take(2000).ToArray();
            TrainOutpus = results.Take(2000).ToArray();
            args.RemoveRange(0, 2000);
            results.RemoveRange(0, 2000);
            ValidationInputs = args.Take(1000).ToArray();
            ValidationOutpus = results.Take(1000).ToArray();

        }

        public InformativeFitness(double[][] features, double [][] results)
        {
            TrainPart = 0.66;
            TrainValidationSplitData(features, results);
        }
        public InformativeFitness(double[][] trainFeatures, double [][] trainResults,double[][] validFeatures, double [][] validResults)
        {
            TrainInputs = trainFeatures;
            TrainOutpus = trainResults;
            ValidationInputs = validFeatures;
            ValidationOutpus = validResults;
        }

        public override double Evaluate(IChromosome chromosome)
        {
            TakeInfornativeFeatures(chromosome);

            
            // create neural network
            ActivationNetwork network = new ActivationNetwork(
                new BipolarSigmoidFunction(1),
                OnlyFeaturesTrainInputs[1].Length,
                20,// two inputs in the network
                10); // one neuron in the second layer

            var stdDevOfWeights = 0.5;
            var randWeights = new GaussianWeights(network, stdDevOfWeights) {UpdateThresholds = true};
            randWeights.Randomize();

            // create teacher
            //network.Layers.Select(x =>
            //    x.Neurons.Select(y => y.Weights.Select(z => (random.NextDouble() - 0.5) * 0.2))); 


            BackPropagationLearning teacher = new BackPropagationLearning(network)
            {
                LearningRate = LearningRate,
                Momentum = Momentum
            };
            // loop
            int teachRuns = MaxEpochs;   //TODO критерии осановки тут


            while (teachRuns > 0)
            {
                // run epoch of learning procedure
                teacher.RunEpoch(OnlyFeaturesTrainInputs, TrainOutpus);
                teachRuns--;
            }
            var errorCounter = 0;
            for (int j = 0; j < OnlyFeaturesTrainInputs.Length; j++)
            {
                network.Compute(OnlyFeaturesTrainInputs[j]);
                if (TrainOutpus[j][network.Output.IndexOf(network.Output.Max())] != 0.6)
                    errorCounter++;
            }
            var errorPart = errorCounter * 100.0 / (TrainOutpus.Length * 1.0);
            Console.WriteLine("learning rate {0}", 100 - errorPart);


            //test data errors
            var errorCounterTest = 0;
            for (int j = 0; j < OnlyFeaturesValidInputs.Length; j++)
            {
                network.Compute(OnlyFeaturesValidInputs[j]);
                if (ValidationOutpus[j][network.Output.IndexOf(network.Output.Max())] != 0.6)
                    errorCounterTest++;
            }
            var errorPartTest = errorCounterTest * 100.0 / (ValidationOutpus.Length * 1.0);
            Console.WriteLine("testing rate {0}", 100 - errorPartTest);



            return 100 - errorPartTest;
        }
        public static void Shuffle(ref double[][] inp, ref double[][] outp)
        {
            var list = new List<InfoContainer>();
            for (int i = 0; i < inp.Length; i++)
            {
                list.Add(new InfoContainer { Inputs = inp[i], Outputs = outp[i] });
            }
            list.Shuffle();
            var sInput = new List<double[]>();
            var sOutput = new List<double[]>();
            foreach (var infoContainer in list)
            {
                sInput.Add(infoContainer.Inputs);
                sOutput.Add(infoContainer.Outputs);
            }
            inp = sInput.ToArray();
            outp = sOutput.ToArray();
        }
    }
    class InfoContainer
    {
        public double[] Inputs { get; set; }
        public double[] Outputs { get; set; }
    }



    public class FeatureSelector
    {
        public ISelectionMethod SelectionMethod { get; set; }
        public IFitnessFunction FitnessFunction { get; set; }
        public int PopulationSize { get; set; } = 100;
        public double MutationRate { get; set; } = 0.1;
        public int LearningRuns { get; set; } = 20;
        public double[][] ImportedData { get; set; }
        public double[][] ImportedResults { get; set; }
        public double TrainPart { get; set; } = 0.66;
        public FeatureSelector(List<string> data)
        {
            PopulationSize = 10;
            SelectionMethod = new EliteSelection();
            FitnessFunction = new InformativeFitness(data);
        }
        public FeatureSelector(double[][] data, double[][] results)
        {
            ImportedData = data;
            ImportedResults = results;
            PopulationSize = 10;
            SelectionMethod = new EliteSelection();
            FitnessFunction = new InformativeFitness(data, results);
            
        }
        public FeatureSelector(int size,ISelectionMethod method,IFitnessFunction fitness)
        {
            PopulationSize = size;
            SelectionMethod = method;
            FitnessFunction = fitness;
        }
        public FeatureSelector(int size,ISelectionMethod method, List<string> data)
        {
            PopulationSize = size;
            SelectionMethod = method;
            FitnessFunction = new InformativeFitness(data);
        }

        public string Select()
        {
            var cromo = new BinaryChromosome(64);
            Console.WriteLine(cromo.ToString());
            string bestChromo = "";

            Population myPopulation = new Population(PopulationSize, cromo, new InformativeFitness(ImportedData, ImportedResults), SelectionMethod) { MutationRate = MutationRate };
            for (int learning = 0; learning < LearningRuns; learning++)
            {
                myPopulation.RunEpoch();
                
                bestChromo = myPopulation.BestChromosome.ToString();

            }

            return bestChromo;
        }
    }

    class FileWorker
    {
        public static List<string> ReadFile(string path)
        {
            var file = new List<string>();
            string line;
            TextReader tr = new StreamReader(path);
            while ((line = tr.ReadLine()) != null)
            {
                file.Add(line);
            }

            return file;
        }

        public static void WriteToFile(List<string> lines, string path)
        {
            using (StreamWriter file =
                new StreamWriter(path))
            {
                foreach (string line in lines)
                {
                    file.WriteLine(line);
                }
            }
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            double[][] data;
            double[][] results;
            FillData(out data, out results);

            //InformativeFitness n = new InformativeFitness();
            
            var feature = new FeatureSelector(data, results);
            var result = feature.Select();

            Console.WriteLine(result);
            int i = 1;
            foreach (var feat in result)
            {
                if(i%8==0)
                    Console.WriteLine(feat == '0' ? ' ' : 'X');
                else
                    Console.Write(feat == '0' ? ' ' : 'X');
                ++i;
            }
            Console.ReadKey();
        }


        public static void FillData(out double[][] features, out double[][] results)
        {
            var data = FileWorker.ReadFile(@"C:\Users\Nick\Desktop\data\data\optdigits.txt");
            data.Shuffle();
            var args = new List<double[]>();
            var ress = new List<double[]>();
            string tmp;
            foreach (var str in data)
            {
                tmp = str.Substring(str.Length - 1);
                var a = new[] { -0.6, -0.6, -0.6, -0.6, -0.6, -0.6, -0.6, -0.6, -0.6, -0.6 }; //10
                var counter = Int32.Parse(tmp);
                a[counter] = 0.6;
                ress.Add(a);
                args.Add(str.Substring(0, str.Length - 2).Split(',')
                    .Select(x => (Double.Parse(x, CultureInfo.InvariantCulture) - 8) / 8).ToArray());
            }
            features = args.ToArray();
            results = ress.ToArray();
        }
    }
}
