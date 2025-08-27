// Copyright John Maddock 2006, 2007
// Copyright Paul A. Bristow 2010

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
using std::cout; using std::endl;
using std::left; using std::fixed; using std::right; using std::scientific;
#include <iomanip>
using std::setw;
using std::setprecision;

#include <boost/math/distributions/chi_squared.hpp>

int error_result = 0;

void confidence_limits_on_std_deviation(
        double Sd,    // Sample Standard Deviation
        unsigned N)   // Sample size
{
   // Calculate confidence intervals for the standard deviation.
   // For example if we set the confidence limit to
   // 0.95, we know that if we repeat the sampling
   // 100 times, then we expect that the true standard deviation
   // will be between out limits on 95 occasions.
   // Note: this is not the same as saying a 95%
   // confidence interval means that there is a 95%
   // probability that the interval contains the true standard deviation.
   // The interval computed from a given sample either
   // contains the true standard deviation or it does not.
   // See http://www.itl.nist.gov/div898/handbook/eda/section3/eda358.htm

   // using namespace boost::math; // potential name ambiguity with std <random>
   using boost::math::chi_squared;
   using boost::math::quantile;
   using boost::math::complement;

   // Print out general info:
   cout <<
      "________________________________________________\n"
      "2-Sided Confidence Limits For Standard Deviation\n"
      "________________________________________________\n\n";
   cout << setprecision(7);
   cout << setw(40) << left << "Number of Observations" << "=  " << N << "\n";
   cout << setw(40) << left << "Standard Deviation" << "=  " << Sd << "\n";
   //
   // Define a table of significance/risk levels:
   double alpha[] = { 0.5, 0.25, 0.1, 0.05, 0.01, 0.001, 0.0001, 0.00001 };
   //
   // Start by declaring the distribution we'll need:
   chi_squared dist(N - 1);
   //
   // Print table header:
   //
   cout << "\n\n"
           "_____________________________________________\n"
           "Confidence          Lower          Upper\n"
           " Value (%)          Limit          Limit\n"
           "_____________________________________________\n";
   //
   // Now print out the data for the table rows.
   for(unsigned i = 0; i < sizeof(alpha)/sizeof(alpha[0]); ++i)
   {
      // Confidence value:
      cout << fixed << setprecision(3) << setw(10) << right << 100 * (1-alpha[i]);
      // Calculate limits:
      double lower_limit = sqrt((N - 1) * Sd * Sd / quantile(complement(dist, alpha[i] / 2)));
      double upper_limit = sqrt((N - 1) * Sd * Sd / quantile(dist, alpha[i] / 2));
      // Print Limits:
      cout << fixed << setprecision(5) << setw(15) << right << lower_limit;
      cout << fixed << setprecision(5) << setw(15) << right << upper_limit << endl;
   }
   cout << endl;
} // void confidence_limits_on_std_deviation

void confidence_limits_on_std_deviation_alpha(
        double Sd,    // Sample Standard Deviation
        double alpha  // confidence
        )
{  // Calculate confidence intervals for the standard deviation.
   // for the alpha parameter, for a range number of observations,
   // from a mere 2 up to a million.
   // O. L. Davies, Statistical Methods in Research and Production, ISBN 0 05 002437 X,
   // 4.33 Page 68, Table H, pp 452 459.

   //   using namespace std;
   // using namespace boost::math;
   using boost::math::chi_squared;
   using boost::math::quantile;
   using boost::math::complement;

   // Define a table of numbers of observations:
   unsigned int obs[] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40 , 50, 60, 100, 120, 1000, 10000, 50000, 100000, 1000000};

   cout <<   // Print out heading:
      "________________________________________________\n"
      "2-Sided Confidence Limits For Standard Deviation\n"
      "________________________________________________\n\n";
   cout << setprecision(7);
   cout << setw(40) << left << "Confidence level (two-sided) " << "=  " << alpha << "\n";
   cout << setw(40) << left << "Standard Deviation" << "=  " << Sd << "\n";

   cout << "\n\n"      // Print table header:
            "_____________________________________________\n"
           "Observations        Lower          Upper\n"
           "                    Limit          Limit\n"
           "_____________________________________________\n";
    for(unsigned i = 0; i < sizeof(obs)/sizeof(obs[0]); ++i)
   {
     unsigned int N = obs[i]; // Observations
     // Start by declaring the distribution with the appropriate :
     chi_squared dist(N - 1);

     // Now print out the data for the table row.
      cout << fixed << setprecision(3) << setw(10) << right << N;
      // Calculate limits: (alpha /2 because it is a two-sided (upper and lower limit) test.
      double lower_limit = sqrt((N - 1) * Sd * Sd / quantile(complement(dist, alpha / 2)));
      double upper_limit = sqrt((N - 1) * Sd * Sd / quantile(dist, alpha / 2));
      // Print Limits:
      cout << fixed << setprecision(4) << setw(15) << right << lower_limit;
      cout << fixed << setprecision(4) << setw(15) << right << upper_limit << endl;
   }
   cout << endl;
}// void confidence_limits_on_std_deviation_alpha

void chi_squared_test(
       double Sd,     // Sample std deviation
       double D,      // True std deviation
       unsigned N,    // Sample size
       double alpha)  // Significance level
{
   //
   // A Chi Squared test applied to a single set of data.
   // We are testing the null hypothesis that the true
   // standard deviation of the sample is D, and that any variation is down
   // to chance.  We can also test the alternative hypothesis
   // that any difference is not down to chance.
   // See http://www.itl.nist.gov/div898/handbook/eda/section3/eda358.htm
   //
   // using namespace boost::math;
   using boost::math::chi_squared;
   using boost::math::quantile;
   using boost::math::complement;
   using boost::math::cdf;

   // Print header:
   cout <<
      "______________________________________________\n"
      "Chi Squared test for sample standard deviation\n"
      "______________________________________________\n\n";
   cout << setprecision(5);
   cout << setw(55) << left << "Number of Observations" << "=  " << N << "\n";
   cout << setw(55) << left << "Sample Standard Deviation" << "=  " << Sd << "\n";
   cout << setw(55) << left << "Expected True Standard Deviation" << "=  " << D << "\n\n";
   //
   // Now we can calculate and output some stats:
   //
   // test-statistic:
   double t_stat = (N - 1) * (Sd / D) * (Sd / D);
   cout << setw(55) << left << "Test Statistic" << "=  " << t_stat << "\n";
   //
   // Finally define our distribution, and get the probability:
   //
   chi_squared dist(N - 1);
   double p = cdf(dist, t_stat);
   cout << setw(55) << left << "CDF of test statistic: " << "=  "
      << setprecision(3) << scientific << p << "\n";
   double ucv = quantile(complement(dist, alpha));
   double ucv2 = quantile(complement(dist, alpha / 2));
   double lcv = quantile(dist, alpha);
   double lcv2 = quantile(dist, alpha / 2);
   cout << setw(55) << left << "Upper Critical Value at alpha: " << "=  "
      << setprecision(3) << scientific << ucv << "\n";
   cout << setw(55) << left << "Upper Critical Value at alpha/2: " << "=  "
      << setprecision(3) << scientific << ucv2 << "\n";
   cout << setw(55) << left << "Lower Critical Value at alpha: " << "=  "
      << setprecision(3) << scientific << lcv << "\n";
   cout << setw(55) << left << "Lower Critical Value at alpha/2: " << "=  "
      << setprecision(3) << scientific << lcv2 << "\n\n";
   //
   // Finally print out results of alternative hypothesis:
   //
   cout << setw(55) << left <<
      "Results for Alternative Hypothesis and alpha" << "=  "
      << setprecision(4) << fixed << alpha << "\n\n";
   cout << "Alternative Hypothesis              Conclusion\n";
   cout << "Standard Deviation != " << setprecision(3) << fixed << D << "            ";
   if((ucv2 < t_stat) || (lcv2 > t_stat))
      cout << "NOT REJECTED\n";
   else
      cout << "REJECTED\n";
   cout << "Standard Deviation  < " << setprecision(3) << fixed << D << "            ";
   if(lcv > t_stat)
      cout << "NOT REJECTED\n";
   else
      cout << "REJECTED\n";
   cout << "Standard Deviation  > " << setprecision(3) << fixed << D << "            ";
   if(ucv < t_stat)
      cout << "NOT REJECTED\n";
   else
      cout << "REJECTED\n";
   cout << endl << endl;
} // void chi_squared_test

void chi_squared_sample_sized(
        double diff,      // difference from variance to detect
        double variance)  // true variance
{
   using namespace std;
   // using boost::math;
   using boost::math::chi_squared;
   using boost::math::quantile;
   using boost::math::complement;
   using boost::math::cdf;

   try
   {
   cout <<   // Print out general info:
     "_____________________________________________________________\n"
      "Estimated sample sizes required for various confidence levels\n"
      "_____________________________________________________________\n\n";
   cout << setprecision(5);
   cout << setw(40) << left << "True Variance" << "=  " << variance << "\n";
   cout << setw(40) << left << "Difference to detect" << "=  " << diff << "\n";
   //
   // Define a table of significance levels:
   //
   double alpha[] = { 0.5, 0.33333333333333333333333, 0.25, 0.1, 0.05, 0.01, 0.001, 0.0001, 0.00001 };
   //
   // Print table header:
   //
   cout << "\n\n"
           "_______________________________________________________________\n"
           "Confidence       Estimated          Estimated\n"
           " Value (%)      Sample Size        Sample Size\n"
           "                (lower one-         (upper one-\n"
           "                 sided test)        sided test)\n"
           "_______________________________________________________________\n";
   //
   // Now print out the data for the table rows.
   //
   for(unsigned i = 0; i < sizeof(alpha)/sizeof(alpha[0]); ++i)
   {
      // Confidence value:
      cout << fixed << setprecision(3) << setw(10) << right << 100 * (1-alpha[i]);
      // Calculate df for a lower single-sided test:
      double df = chi_squared::find_degrees_of_freedom(
         -diff, alpha[i], alpha[i], variance);
      // Convert to integral sample size (df is a floating point value in this implementation):
      double size = ceil(df) + 1;
      // Print size:
      cout << fixed << setprecision(0) << setw(16) << right << size;
      // Calculate df for an upper single-sided test:
      df = chi_squared::find_degrees_of_freedom(
         diff, alpha[i], alpha[i], variance);
      // Convert to integral sample size:
      size = ceil(df) + 1;
      // Print size:
      cout << fixed << setprecision(0) << setw(16) << right << size << endl;
   }
   cout << endl;
   }
  catch(const std::exception& e)
  { // Always useful to include try & catch blocks because default policies
    // are to throw exceptions on arguments that cause errors like underflow, overflow.
    // Lacking try & catch blocks, the program will abort without a message below,
    // which may give some helpful clues as to the cause of the exception.
    std::cout <<
      "\n""Message from thrown exception was:\n   " << e.what() << std::endl;
    ++error_result;
  }
} // chi_squared_sample_sized

int main()
{
   // Run tests for Gear data
   // see http://www.itl.nist.gov/div898/handbook/eda/section3/eda3581.htm
   // Tests measurements of gear diameter.
   //
   confidence_limits_on_std_deviation(0.6278908E-02, 100);
   chi_squared_test(0.6278908E-02, 0.1, 100, 0.05);
   chi_squared_sample_sized(0.1 - 0.6278908E-02, 0.1);
   //
   // Run tests for silicon wafer fabrication data.
   // see http://www.itl.nist.gov/div898/handbook/prc/section2/prc23.htm
   // A supplier of 100 ohm.cm silicon wafers claims that his fabrication
   // process can produce wafers with sufficient consistency so that the
   // standard deviation of resistivity for the lot does not exceed
   // 10 ohm.cm. A sample of N = 10 wafers taken from the lot has a
   // standard deviation of 13.97 ohm.cm
   //
   confidence_limits_on_std_deviation(13.97, 10);
   chi_squared_test(13.97, 10.0, 10, 0.05);
   chi_squared_sample_sized(13.97 * 13.97 - 100, 100);
   chi_squared_sample_sized(55, 100);
   chi_squared_sample_sized(1, 100);

   // List confidence interval multipliers for standard deviation
   // for a range of numbers of observations from 2 to a million,
   // and for a few alpha values, 0.1, 0.05, 0.01 for confidences 90, 95, 99 %
   confidence_limits_on_std_deviation_alpha(1., 0.1);
   confidence_limits_on_std_deviation_alpha(1., 0.05);
   confidence_limits_on_std_deviation_alpha(1., 0.01);

   return error_result;
}

/*

________________________________________________
2-Sided Confidence Limits For Standard Deviation
________________________________________________
Number of Observations                  =  100
Standard Deviation                      =  0.006278908
_____________________________________________
Confidence          Lower          Upper
 Value (%)          Limit          Limit
_____________________________________________
    50.000        0.00601        0.00662
    75.000        0.00582        0.00685
    90.000        0.00563        0.00712
    95.000        0.00551        0.00729
    99.000        0.00530        0.00766
    99.900        0.00507        0.00812
    99.990        0.00489        0.00855
    99.999        0.00474        0.00895
______________________________________________
Chi Squared test for sample standard deviation
______________________________________________
Number of Observations                                 =  100
Sample Standard Deviation                              =  0.00628
Expected True Standard Deviation                       =  0.10000
Test Statistic                                         =  0.39030
CDF of test statistic:                                 =  1.438e-099
Upper Critical Value at alpha:                         =  1.232e+002
Upper Critical Value at alpha/2:                       =  1.284e+002
Lower Critical Value at alpha:                         =  7.705e+001
Lower Critical Value at alpha/2:                       =  7.336e+001
Results for Alternative Hypothesis and alpha           =  0.0500
Alternative Hypothesis              Conclusion
Standard Deviation != 0.100            NOT REJECTED
Standard Deviation  < 0.100            NOT REJECTED
Standard Deviation  > 0.100            REJECTED
_____________________________________________________________
Estimated sample sizes required for various confidence levels
_____________________________________________________________
True Variance                           =  0.10000
Difference to detect                    =  0.09372
_______________________________________________________________
Confidence       Estimated          Estimated
 Value (%)      Sample Size        Sample Size
                (lower one-         (upper one-
                 sided test)        sided test)
_______________________________________________________________
    50.000               2               2
    66.667               2               5
    75.000               2              10
    90.000               4              32
    95.000               5              52
    99.000               8             102
    99.900              13             178
    99.990              18             257
    99.999              23             337
________________________________________________
2-Sided Confidence Limits For Standard Deviation
________________________________________________
Number of Observations                  =  10
Standard Deviation                      =  13.9700000
_____________________________________________
Confidence          Lower          Upper
 Value (%)          Limit          Limit
_____________________________________________
    50.000       12.41880       17.25579
    75.000       11.23084       19.74131
    90.000       10.18898       22.98341
    95.000        9.60906       25.50377
    99.000        8.62898       31.81825
    99.900        7.69466       42.51593
    99.990        7.04085       55.93352
    99.999        6.54517       73.00132
______________________________________________
Chi Squared test for sample standard deviation
______________________________________________
Number of Observations                                 =  10
Sample Standard Deviation                              =  13.97000
Expected True Standard Deviation                       =  10.00000
Test Statistic                                         =  17.56448
CDF of test statistic:                                 =  9.594e-001
Upper Critical Value at alpha:                         =  1.692e+001
Upper Critical Value at alpha/2:                       =  1.902e+001
Lower Critical Value at alpha:                         =  3.325e+000
Lower Critical Value at alpha/2:                       =  2.700e+000
Results for Alternative Hypothesis and alpha           =  0.0500
Alternative Hypothesis              Conclusion
Standard Deviation != 10.000            REJECTED
Standard Deviation  < 10.000            REJECTED
Standard Deviation  > 10.000            NOT REJECTED
_____________________________________________________________
Estimated sample sizes required for various confidence levels
_____________________________________________________________
True Variance                           =  100.00000
Difference to detect                    =  95.16090
_______________________________________________________________
Confidence       Estimated          Estimated
 Value (%)      Sample Size        Sample Size
                (lower one-         (upper one-
                 sided test)        sided test)
_______________________________________________________________
    50.000               2               2
    66.667               2               5
    75.000               2              10
    90.000               4              32
    95.000               5              51
    99.000               7              99
    99.900              11             174
    99.990              15             251
    99.999              20             330
_____________________________________________________________
Estimated sample sizes required for various confidence levels
_____________________________________________________________
True Variance                           =  100.00000
Difference to detect                    =  55.00000
_______________________________________________________________
Confidence       Estimated          Estimated
 Value (%)      Sample Size        Sample Size
                (lower one-         (upper one-
                 sided test)        sided test)
_______________________________________________________________
    50.000               2               2
    66.667               4              10
    75.000               8              21
    90.000              23              71
    95.000              36             115
    99.000              71             228
    99.900             123             401
    99.990             177             580
    99.999             232             762
_____________________________________________________________
Estimated sample sizes required for various confidence levels
_____________________________________________________________
True Variance                           =  100.00000
Difference to detect                    =  1.00000
_______________________________________________________________
Confidence       Estimated          Estimated
 Value (%)      Sample Size        Sample Size
                (lower one-         (upper one-
                 sided test)        sided test)
_______________________________________________________________
    50.000               2               2
    66.667           14696           14993
    75.000           36033           36761
    90.000          130079          132707
    95.000          214283          218612
    99.000          428628          437287
    99.900          756333          771612
    99.990         1095435         1117564
    99.999         1440608         1469711
________________________________________________
2-Sided Confidence Limits For Standard Deviation
________________________________________________
Confidence level (two-sided)            =  0.1000000
Standard Deviation                      =  1.0000000
_____________________________________________
Observations        Lower          Upper
                    Limit          Limit
_____________________________________________
         2         0.5102        15.9472
         3         0.5778         4.4154
         4         0.6196         2.9200
         5         0.6493         2.3724
         6         0.6720         2.0893
         7         0.6903         1.9154
         8         0.7054         1.7972
         9         0.7183         1.7110
        10         0.7293         1.6452
        15         0.7688         1.4597
        20         0.7939         1.3704
        30         0.8255         1.2797
        40         0.8454         1.2320
        50         0.8594         1.2017
        60         0.8701         1.1805
       100         0.8963         1.1336
       120         0.9045         1.1203
      1000         0.9646         1.0383
     10000         0.9885         1.0118
     50000         0.9948         1.0052
    100000         0.9963         1.0037
   1000000         0.9988         1.0012
________________________________________________
2-Sided Confidence Limits For Standard Deviation
________________________________________________
Confidence level (two-sided)            =  0.0500000
Standard Deviation                      =  1.0000000
_____________________________________________
Observations        Lower          Upper
                    Limit          Limit
_____________________________________________
         2         0.4461        31.9102
         3         0.5207         6.2847
         4         0.5665         3.7285
         5         0.5991         2.8736
         6         0.6242         2.4526
         7         0.6444         2.2021
         8         0.6612         2.0353
         9         0.6755         1.9158
        10         0.6878         1.8256
        15         0.7321         1.5771
        20         0.7605         1.4606
        30         0.7964         1.3443
        40         0.8192         1.2840
        50         0.8353         1.2461
        60         0.8476         1.2197
       100         0.8780         1.1617
       120         0.8875         1.1454
      1000         0.9580         1.0459
     10000         0.9863         1.0141
     50000         0.9938         1.0062
    100000         0.9956         1.0044
   1000000         0.9986         1.0014
________________________________________________
2-Sided Confidence Limits For Standard Deviation
________________________________________________
Confidence level (two-sided)            =  0.0100000
Standard Deviation                      =  1.0000000
_____________________________________________
Observations        Lower          Upper
                    Limit          Limit
_____________________________________________
         2         0.3562       159.5759
         3         0.4344        14.1244
         4         0.4834         6.4675
         5         0.5188         4.3960
         6         0.5464         3.4848
         7         0.5688         2.9798
         8         0.5875         2.6601
         9         0.6036         2.4394
        10         0.6177         2.2776
        15         0.6686         1.8536
        20         0.7018         1.6662
        30         0.7444         1.4867
        40         0.7718         1.3966
        50         0.7914         1.3410
        60         0.8065         1.3026
       100         0.8440         1.2200
       120         0.8558         1.1973
      1000         0.9453         1.0609
     10000         0.9821         1.0185
     50000         0.9919         1.0082
    100000         0.9943         1.0058
   1000000         0.9982         1.0018
*/

