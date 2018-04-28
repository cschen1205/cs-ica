using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MathNet.Numerics.LinearAlgebra.Generic;
using MathNet.Numerics.LinearAlgebra.Double;

namespace SimuKit.ML.ICA
{
    /// <summary>
    /// Independent Component Analysis algorithm for solving linear noise less "cocktail party problem"
    /// Example: Blind Source Separation of recorded speech and music signals (a.k.a Cocktail Party Problem)
    /// </summary>
    public class LinearNoiseLessICA
    {
        /// <summary>
        /// Separate the original signal sample X into two independent sources X1 and X2
        /// </summary>
        /// <param name="X"></param>
        /// <param name="X1"></param>
        /// <param name="X2"></param>
        public void Solve(List<double[]> X, out List<double[]> X1, out List<double[]> X2)
        {
            int m=X.Count;
            if(m==0)
            {
                throw new Exception("sample count is zero!");
            }

            double[] signal_0 = X[0];
            int num_features=signal_0.Length;

            Matrix<double> X_matrix = new DenseMatrix(m, num_features);

            for (int i = 0; i < m; ++i)
            {
                double[] rec=X[i];
                for (int j = 0; j < num_features; ++j)
                {
                    X_matrix[i, j] = rec[j];
                }
            }

            Matrix<double> Xp2_matrix = X_matrix.PointwiseMultiply(X_matrix);

            double[] X_sum_1=new double[num_features];
            for (int j = 0; j < num_features; ++j)
            {
                double sum = 0;
                for(int i=0; i < m; ++i)
                {
                    sum += Xp2_matrix[i, j];
                }
                X_sum_1[j] = sum;
            }

            Matrix<double> X_repmat = new DenseMatrix(m, num_features);

            for (int j = 0; j < num_features; ++j)
            {
                double num = X_sum_1[j];
                for (int i = 0; i < m; ++i)
                {
                    X_repmat[i, j] = num;
                }
            }

            Matrix<double> X_repmat2 = X_repmat.PointwiseMultiply(X_matrix);
            Matrix<double> X_Cov = X_repmat2.Multiply(X_matrix.Transpose());

            var svd=X_Cov.Svd(true);
            Matrix<double> W=svd.U();
            Vector<double> s = svd.S();
            Matrix<double> v = svd.VT();

            Matrix<double> X1_matrix=W.Multiply(X_matrix);
            Matrix<double> X2_matrix=v.Multiply(X_matrix);

            X1 = new List<double[]>();
            X2 = new List<double[]>();

            for (int i = 0; i < m; ++i)
            {
                double[] signal1 = Clone(signal_0);
                double[] signal2 = Clone(signal_0);
                for (int j = 0; j < num_features; ++j)
                {
                    signal1[j] = X1_matrix[i, j];
                    signal2[j] = X2_matrix[i, j];
                }
                X1.Add(signal1);
                X2.Add(signal2);
            }
        }

        private double[] Clone(double[] a)
        {
            double[] result = new double[a.Length];
            for(int i=0; i <a.Length; ++i)
            {
                result[i] = a[i];
            }
            return result;
        }
    }


}
