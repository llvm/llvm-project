// Copyright John Maddock 2006
// Copyright Paul A. Bristow 2007, 2010

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifdef _MSC_VER
#  pragma warning(disable: 4512) // assignment operator could not be generated.
#  pragma warning(disable: 4510) // default constructor could not be generated.
#  pragma warning(disable: 4610) // can never be instantiated - user defined constructor required.
#endif

#include <boost/math/distributions/students_t.hpp>

// avoid "using namespace std;" and "using namespace boost::math;"
// to avoid potential ambiguity with names in std random.
#include <iostream>
using std::cout; using std::endl;
using std::left; using std::fixed; using std::right; using std::scientific;
#include <iomanip>
using std::setw;
using std::setprecision;

void confidence_limits_on_mean(double Sm, double Sd, unsigned Sn)
{
   //
   // Sm = Sample Mean.
   // Sd = Sample Standard Deviation.
   // Sn = Sample Size.
   //
   // Calculate confidence intervals for the mean.
   // For example if we set the confidence limit to
   // 0.95, we know that if we repeat the sampling
   // 100 times, then we expect that the true mean
   // will be between out limits on 95 occasions.
   // Note: this is not the same as saying a 95%
   // confidence interval means that there is a 95%
   // probability that the interval contains the true mean.
   // The interval computed from a given sample either
   // contains the true mean or it does not.
   // See http://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm

   using boost::math::students_t;

   // Print out general info:
   cout <<
      "__________________________________\n"
      "2-Sided Confidence Limits For Mean\n"
      "__________________________________\n\n";
   cout << setprecision(7);
   cout << setw(40) << left << "Number of Observations" << "=  " << Sn << "\n";
   cout << setw(40) << left << "Mean" << "=  " << Sm << "\n";
   cout << setw(40) << left << "Standard Deviation" << "=  " << Sd << "\n";
   //
   // Define a table of significance/risk levels:
   //
   double alpha[] = { 0.5, 0.25, 0.1, 0.05, 0.01, 0.001, 0.0001, 0.00001 };
   //
   // Start by declaring the distribution we'll need:
   //
   students_t dist(Sn - 1);
   //
   // Print table header:
   //
   cout << "\n\n"
           "_______________________________________________________________\n"
           "Confidence       T           Interval          Lower          Upper\n"
           " Value (%)     Value          Width            Limit          Limit\n"
           "_______________________________________________________________\n";
   //
   // Now print out the data for the table rows.
   //
   for(unsigned i = 0; i < sizeof(alpha)/sizeof(alpha[0]); ++i)
   {
      // Confidence value:
      cout << fixed << setprecision(3) << setw(10) << right << 100 * (1-alpha[i]);
      // calculate T:
      double T = quantile(complement(dist, alpha[i] / 2));
      // Print T:
      cout << fixed << setprecision(3) << setw(10) << right << T;
      // Calculate width of interval (one sided):
      double w = T * Sd / sqrt(double(Sn));
      // Print width:
      if(w < 0.01)
         cout << scientific << setprecision(3) << setw(17) << right << w;
      else
         cout << fixed << setprecision(3) << setw(17) << right << w;
      // Print Limits:
      cout << fixed << setprecision(5) << setw(15) << right << Sm - w;
      cout << fixed << setprecision(5) << setw(15) << right << Sm + w << endl;
   }
   cout << endl;
} // void confidence_limits_on_mean

void single_sample_t_test(double M, double Sm, double Sd, unsigned Sn, double alpha)
{
   //
   // M = true mean.
   // Sm = Sample Mean.
   // Sd = Sample Standard Deviation.
   // Sn = Sample Size.
   // alpha = Significance Level.
   //
   // A Students t test applied to a single set of data.
   // We are testing the null hypothesis that the true
   // mean of the sample is M, and that any variation is down
   // to chance.  We can also test the alternative hypothesis
   // that any difference is not down to chance.
   // See http://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm
   
   using boost::math::students_t;

   // Print header:
   cout <<
      "__________________________________\n"
      "Student t test for a single sample\n"
      "__________________________________\n\n";
   cout << setprecision(5);
   cout << setw(55) << left << "Number of Observations" << "=  " << Sn << "\n";
   cout << setw(55) << left << "Sample Mean" << "=  " << Sm << "\n";
   cout << setw(55) << left << "Sample Standard Deviation" << "=  " << Sd << "\n";
   cout << setw(55) << left << "Expected True Mean" << "=  " << M << "\n\n";
   //
   // Now we can calculate and output some stats:
   //
   // Difference in means:
   double diff = Sm - M;
   cout << setw(55) << left << "Sample Mean - Expected Test Mean" << "=  " << diff << "\n";
   // Degrees of freedom:
   unsigned v = Sn - 1;
   cout << setw(55) << left << "Degrees of Freedom" << "=  " << v << "\n";
   // t-statistic:
   double t_stat = diff * sqrt(double(Sn)) / Sd;
   cout << setw(55) << left << "T Statistic" << "=  " << t_stat << "\n";
   //
   // Finally define our distribution, and get the probability:
   //
   students_t dist(v);
   double q = cdf(complement(dist, fabs(t_stat)));
   cout << setw(55) << left << "Probability that difference is due to chance" << "=  "
      << setprecision(3) << scientific << 2 * q << "\n\n";
   //
   // Finally print out results of alternative hypothesis:
   //
   cout << setw(55) << left <<
      "Results for Alternative Hypothesis and alpha" << "=  "
      << setprecision(4) << fixed << alpha << "\n\n";
   cout << "Alternative Hypothesis     Conclusion\n";
   cout << "Mean != " << setprecision(3) << fixed << M << "            ";
   if(q < alpha / 2)
      cout << "NOT REJECTED\n";
   else
      cout << "REJECTED\n";
   cout << "Mean  < " << setprecision(3) << fixed << M << "            ";
   if(cdf(complement(dist, t_stat)) > alpha)
      cout << "NOT REJECTED\n";
   else
      cout << "REJECTED\n";
   cout << "Mean  > " << setprecision(3) << fixed << M << "            ";
   if(cdf(dist, t_stat) > alpha)
      cout << "NOT REJECTED\n";
   else
      cout << "REJECTED\n";
   cout << endl << endl;
} // void single_sample_t_test(

void single_sample_find_df(double M, double Sm, double Sd)
{
   //
   // M = true mean.
   // Sm = Sample Mean.
   // Sd = Sample Standard Deviation.
   //
 
   using boost::math::students_t;

   // Print out general info:
   cout <<
      "_____________________________________________________________\n"
      "Estimated sample sizes required for various confidence levels\n"
      "_____________________________________________________________\n\n";
   cout << setprecision(5);
   cout << setw(40) << left << "True Mean" << "=  " << M << "\n";
   cout << setw(40) << left << "Sample Mean" << "=  " << Sm << "\n";
   cout << setw(40) << left << "Sample Standard Deviation" << "=  " << Sd << "\n";
   //
   // Define a table of significance intervals:
   //
   double alpha[] = { 0.5, 0.25, 0.1, 0.05, 0.01, 0.001, 0.0001, 0.00001 };
   //
   // Print table header:
   //
   cout << "\n\n"
           "_______________________________________________________________\n"
           "Confidence       Estimated          Estimated\n"
           " Value (%)      Sample Size        Sample Size\n"
           "              (one sided test)    (two sided test)\n"
           "_______________________________________________________________\n";
   //
   // Now print out the data for the table rows.
   //
   for(unsigned i = 1; i < sizeof(alpha)/sizeof(alpha[0]); ++i)
   {
      // Confidence value:
      cout << fixed << setprecision(3) << setw(10) << right << 100 * (1-alpha[i]);
      // calculate df for single sided test:
      double df = students_t::find_degrees_of_freedom(
         fabs(M - Sm), alpha[i], alpha[i], Sd);
      // convert to sample size, always one more than the degrees of freedom:
      double size = ceil(df) + 1;
      // Print size:
      cout << fixed << setprecision(0) << setw(16) << right << size;
      // calculate df for two sided test:
      df = students_t::find_degrees_of_freedom(
         fabs(M - Sm), alpha[i]/2, alpha[i], Sd);
      // convert to sample size:
      size = ceil(df) + 1;
      // Print size:
      cout << fixed << setprecision(0) << setw(16) << right << size << endl;
   }
   cout << endl;
} // void single_sample_find_df

int main()
{
   //
   // Run tests for Heat Flow Meter data
   // see http://www.itl.nist.gov/div898/handbook/eda/section4/eda428.htm
   // The data was collected while calibrating a heat flow meter
   // against a known value.
   //
   confidence_limits_on_mean(9.261460, 0.2278881e-01, 195);
   single_sample_t_test(5, 9.261460, 0.2278881e-01, 195, 0.05);
   single_sample_find_df(5, 9.261460, 0.2278881e-01);

   //
   // Data for this example from:
   // P.K.Hou, O. W. Lau & M.C. Wong, Analyst (1983) vol. 108, p 64.
   // from Statistics for Analytical Chemistry, 3rd ed. (1994), pp 54-55
   // J. C. Miller and J. N. Miller, Ellis Horwood ISBN 0 13 0309907
   //
   // Determination of mercury by cold-vapour atomic absorption,
   // the following values were obtained fusing a trusted
   // Standard Reference Material containing 38.9% mercury,
   // which we assume is correct or 'true'.
   //
   confidence_limits_on_mean(37.8, 0.964365, 3);
   // 95% test:
   single_sample_t_test(38.9, 37.8, 0.964365, 3, 0.05);
   // 90% test:
   single_sample_t_test(38.9, 37.8, 0.964365, 3, 0.1);
   // parameter estimate:
   single_sample_find_df(38.9, 37.8, 0.964365);

   return 0;
} // int main()

/*

Output:

------ Rebuild All started: Project: students_t_single_sample, Configuration: Release Win32 ------
  students_t_single_sample.cpp
  Generating code
  Finished generating code
  students_t_single_sample.vcxproj -> J:\Cpp\MathToolkit\test\Math_test\Release\students_t_single_sample.exe
__________________________________
2-Sided Confidence Limits For Mean
__________________________________

Number of Observations                  =  195
Mean                                    =  9.26146
Standard Deviation                      =  0.02278881


_______________________________________________________________
Confidence       T           Interval          Lower          Upper
 Value (%)     Value          Width            Limit          Limit
_______________________________________________________________
    50.000     0.676       1.103e-003        9.26036        9.26256
    75.000     1.154       1.883e-003        9.25958        9.26334
    90.000     1.653       2.697e-003        9.25876        9.26416
    95.000     1.972       3.219e-003        9.25824        9.26468
    99.000     2.601       4.245e-003        9.25721        9.26571
    99.900     3.341       5.453e-003        9.25601        9.26691
    99.990     3.973       6.484e-003        9.25498        9.26794
    99.999     4.537       7.404e-003        9.25406        9.26886

__________________________________
Student t test for a single sample
__________________________________

Number of Observations                                 =  195
Sample Mean                                            =  9.26146
Sample Standard Deviation                              =  0.02279
Expected True Mean                                     =  5.00000

Sample Mean - Expected Test Mean                       =  4.26146
Degrees of Freedom                                     =  194
T Statistic                                            =  2611.28380
Probability that difference is due to chance           =  0.000e+000

Results for Alternative Hypothesis and alpha           =  0.0500

Alternative Hypothesis     Conclusion
Mean != 5.000            NOT REJECTED
Mean  < 5.000            REJECTED
Mean  > 5.000            NOT REJECTED


_____________________________________________________________
Estimated sample sizes required for various confidence levels
_____________________________________________________________

True Mean                               =  5.00000
Sample Mean                             =  9.26146
Sample Standard Deviation               =  0.02279


_______________________________________________________________
Confidence       Estimated          Estimated
 Value (%)      Sample Size        Sample Size
              (one sided test)    (two sided test)
_______________________________________________________________
    75.000               2               2
    90.000               2               2
    95.000               2               2
    99.000               2               2
    99.900               3               3
    99.990               3               3
    99.999               4               4

__________________________________
2-Sided Confidence Limits For Mean
__________________________________

Number of Observations                  =  3
Mean                                    =  37.8000000
Standard Deviation                      =  0.9643650


_______________________________________________________________
Confidence       T           Interval          Lower          Upper
 Value (%)     Value          Width            Limit          Limit
_______________________________________________________________
    50.000     0.816            0.455       37.34539       38.25461
    75.000     1.604            0.893       36.90717       38.69283
    90.000     2.920            1.626       36.17422       39.42578
    95.000     4.303            2.396       35.40438       40.19562
    99.000     9.925            5.526       32.27408       43.32592
    99.900    31.599           17.594       20.20639       55.39361
    99.990    99.992           55.673      -17.87346       93.47346
    99.999   316.225          176.067     -138.26683      213.86683

__________________________________
Student t test for a single sample
__________________________________

Number of Observations                                 =  3
Sample Mean                                            =  37.80000
Sample Standard Deviation                              =  0.96437
Expected True Mean                                     =  38.90000

Sample Mean - Expected Test Mean                       =  -1.10000
Degrees of Freedom                                     =  2
T Statistic                                            =  -1.97566
Probability that difference is due to chance           =  1.869e-001

Results for Alternative Hypothesis and alpha           =  0.0500

Alternative Hypothesis     Conclusion
Mean != 38.900            REJECTED
Mean  < 38.900            NOT REJECTED
Mean  > 38.900            NOT REJECTED


__________________________________
Student t test for a single sample
__________________________________

Number of Observations                                 =  3
Sample Mean                                            =  37.80000
Sample Standard Deviation                              =  0.96437
Expected True Mean                                     =  38.90000

Sample Mean - Expected Test Mean                       =  -1.10000
Degrees of Freedom                                     =  2
T Statistic                                            =  -1.97566
Probability that difference is due to chance           =  1.869e-001

Results for Alternative Hypothesis and alpha           =  0.1000

Alternative Hypothesis     Conclusion
Mean != 38.900            REJECTED
Mean  < 38.900            NOT REJECTED
Mean  > 38.900            REJECTED


_____________________________________________________________
Estimated sample sizes required for various confidence levels
_____________________________________________________________

True Mean                               =  38.90000
Sample Mean                             =  37.80000
Sample Standard Deviation               =  0.96437


_______________________________________________________________
Confidence       Estimated          Estimated
 Value (%)      Sample Size        Sample Size
              (one sided test)    (two sided test)
_______________________________________________________________
    75.000               3               4
    90.000               7               9
    95.000              11              13
    99.000              20              22
    99.900              35              37
    99.990              50              53
    99.999              66              68

*/

