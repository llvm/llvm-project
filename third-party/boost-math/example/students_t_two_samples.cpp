// Copyright John Maddock 2006.
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

#include <iostream>
using std::cout; using std::endl;
using std::left; using std::fixed; using std::right; using std::scientific;
#include <iomanip>
using std::setw;
using std::setprecision;

#include <boost/math/distributions/students_t.hpp>
   using boost::math::students_t;


void two_samples_t_test_equal_sd(
        double Sm1, // Sm1 = Sample Mean 1.
        double Sd1,   // Sd1 = Sample Standard Deviation 1.
        unsigned Sn1,   // Sn1 = Sample Size 1.
        double Sm2,   // Sm2 = Sample Mean 2.
        double Sd2,   // Sd2 = Sample Standard Deviation 2.
        unsigned Sn2,   // Sn2 = Sample Size 2.
        double alpha)   // alpha = Significance Level.
{
   // A Students t test applied to two sets of data.
   // We are testing the null hypothesis that the two
   // samples have the same mean and that any difference
   // if due to chance.
   // See http://www.itl.nist.gov/div898/handbook/eda/section3/eda353.htm
   //
   using namespace std;
   // using namespace boost::math;

   using boost::math::students_t;

   // Print header:
   cout <<
      "_______________________________________________\n"
      "Student t test for two samples (equal variances)\n"
      "_______________________________________________\n\n";
   cout << setprecision(5);
   cout << setw(55) << left << "Number of Observations (Sample 1)" << "=  " << Sn1 << "\n";
   cout << setw(55) << left << "Sample 1 Mean" << "=  " << Sm1 << "\n";
   cout << setw(55) << left << "Sample 1 Standard Deviation" << "=  " << Sd1 << "\n";
   cout << setw(55) << left << "Number of Observations (Sample 2)" << "=  " << Sn2 << "\n";
   cout << setw(55) << left << "Sample 2 Mean" << "=  " << Sm2 << "\n";
   cout << setw(55) << left << "Sample 2 Standard Deviation" << "=  " << Sd2 << "\n";
   //
   // Now we can calculate and output some stats:
   //
   // Degrees of freedom:
   double v = Sn1 + Sn2 - 2;
   cout << setw(55) << left << "Degrees of Freedom" << "=  " << v << "\n";
   // Pooled variance and hence standard deviation:
   double sp = sqrt(((Sn1-1) * Sd1 * Sd1 + (Sn2-1) * Sd2 * Sd2) / v);
   cout << setw(55) << left << "Pooled Standard Deviation" << "=  " << sp << "\n";
   // t-statistic:
   double t_stat = (Sm1 - Sm2) / (sp * sqrt(1.0 / Sn1 + 1.0 / Sn2));
   cout << setw(55) << left << "T Statistic" << "=  " << t_stat << "\n";
   //
   // Define our distribution, and get the probability:
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
   cout << "Alternative Hypothesis              Conclusion\n";
   cout << "Sample 1 Mean != Sample 2 Mean       " ;
   if(q < alpha / 2)
      cout << "NOT REJECTED\n";
   else
      cout << "REJECTED\n";
   cout << "Sample 1 Mean <  Sample 2 Mean       ";
   if(cdf(dist, t_stat) < alpha)
      cout << "NOT REJECTED\n";
   else
      cout << "REJECTED\n";
   cout << "Sample 1 Mean >  Sample 2 Mean       ";
   if(cdf(complement(dist, t_stat)) < alpha)
      cout << "NOT REJECTED\n";
   else
      cout << "REJECTED\n";
   cout << endl << endl;
}

void two_samples_t_test_unequal_sd(
        double Sm1,   // Sm1 = Sample Mean 1.
        double Sd1,   // Sd1 = Sample Standard Deviation 1.
        unsigned Sn1,   // Sn1 = Sample Size 1.
        double Sm2,   // Sm2 = Sample Mean 2.
        double Sd2,   // Sd2 = Sample Standard Deviation 2.
        unsigned Sn2,   // Sn2 = Sample Size 2.
        double alpha)   // alpha = Significance Level.
{
   // A Students t test applied to two sets of data.
   // We are testing the null hypothesis that the two
   // samples have the same mean and 
   // that any difference is due to chance.
   // See http://www.itl.nist.gov/div898/handbook/eda/section3/eda353.htm
   //
   using namespace std;
   using boost::math::students_t;

   // Print header:
   cout <<
      "_________________________________________________\n"
      "Student t test for two samples (unequal variances)\n"
      "_________________________________________________\n\n";
   cout << setprecision(5);
   cout << setw(55) << left << "Number of Observations (Sample 1)" << "=  " << Sn1 << "\n";
   cout << setw(55) << left << "Sample 1 Mean" << "=  " << Sm1 << "\n";
   cout << setw(55) << left << "Sample 1 Standard Deviation" << "=  " << Sd1 << "\n";
   cout << setw(55) << left << "Number of Observations (Sample 2)" << "=  " << Sn2 << "\n";
   cout << setw(55) << left << "Sample 2 Mean" << "=  " << Sm2 << "\n";
   cout << setw(55) << left << "Sample 2 Standard Deviation" << "=  " << Sd2 << "\n";
   //
   // Now we can calculate and output some stats:
   //
   // Degrees of freedom:
   double v = Sd1 * Sd1 / Sn1 + Sd2 * Sd2 / Sn2;
   v *= v;
   double t1 = Sd1 * Sd1 / Sn1;
   t1 *= t1;
   t1 /=  (Sn1 - 1);
   double t2 = Sd2 * Sd2 / Sn2;
   t2 *= t2;
   t2 /= (Sn2 - 1);
   v /= (t1 + t2);
   cout << setw(55) << left << "Degrees of Freedom" << "=  " << v << "\n";
   // t-statistic:
   double t_stat = (Sm1 - Sm2) / sqrt(Sd1 * Sd1 / Sn1 + Sd2 * Sd2 / Sn2);
   cout << setw(55) << left << "T Statistic" << "=  " << t_stat << "\n";
   //
   // Define our distribution, and get the probability:
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
   cout << "Alternative Hypothesis              Conclusion\n";
   cout << "Sample 1 Mean != Sample 2 Mean       " ;
   if(q < alpha / 2)
      cout << "NOT REJECTED\n";
   else
      cout << "REJECTED\n";
   cout << "Sample 1 Mean <  Sample 2 Mean       ";
   if(cdf(dist, t_stat) < alpha)
      cout << "NOT REJECTED\n";
   else
      cout << "REJECTED\n";
   cout << "Sample 1 Mean >  Sample 2 Mean       ";
   if(cdf(complement(dist, t_stat)) < alpha)
      cout << "NOT REJECTED\n";
   else
      cout << "REJECTED\n";
   cout << endl << endl;
}

int main()
{
   //
   // Run tests for Car Mileage sample data
   // http://www.itl.nist.gov/div898/handbook/eda/section3/eda3531.htm
   // from the NIST website http://www.itl.nist.gov.  The data compares
   // miles per gallon of US cars with miles per gallon of Japanese cars.
   //
   two_samples_t_test_equal_sd(20.14458, 6.414700, 249, 30.48101, 6.107710, 79, 0.05);
   two_samples_t_test_unequal_sd(20.14458, 6.414700, 249, 30.48101, 6.107710, 79, 0.05);

   return 0;
} // int main()

/*
Output is:

  _______________________________________________
  Student t test for two samples (equal variances)
  _______________________________________________
  
  Number of Observations (Sample 1)                      =  249
  Sample 1 Mean                                          =  20.145
  Sample 1 Standard Deviation                            =  6.4147
  Number of Observations (Sample 2)                      =  79
  Sample 2 Mean                                          =  30.481
  Sample 2 Standard Deviation                            =  6.1077
  Degrees of Freedom                                     =  326
  Pooled Standard Deviation                              =  6.3426
  T Statistic                                            =  -12.621
  Probability that difference is due to chance           =  5.273e-030
  
  Results for Alternative Hypothesis and alpha           =  0.0500
  
  Alternative Hypothesis              Conclusion
  Sample 1 Mean != Sample 2 Mean       NOT REJECTED
  Sample 1 Mean <  Sample 2 Mean       NOT REJECTED
  Sample 1 Mean >  Sample 2 Mean       REJECTED
  
  
  _________________________________________________
  Student t test for two samples (unequal variances)
  _________________________________________________
  
  Number of Observations (Sample 1)                      =  249
  Sample 1 Mean                                          =  20.14458
  Sample 1 Standard Deviation                            =  6.41470
  Number of Observations (Sample 2)                      =  79
  Sample 2 Mean                                          =  30.48101
  Sample 2 Standard Deviation                            =  6.10771
  Degrees of Freedom                                     =  136.87499
  T Statistic                                            =  -12.94627
  Probability that difference is due to chance           =  1.571e-025
  
  Results for Alternative Hypothesis and alpha           =  0.0500
  
  Alternative Hypothesis              Conclusion
  Sample 1 Mean != Sample 2 Mean       NOT REJECTED
  Sample 1 Mean <  Sample 2 Mean       NOT REJECTED
  Sample 1 Mean >  Sample 2 Mean       REJECTED
  


*/

