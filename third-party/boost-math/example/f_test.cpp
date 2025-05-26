// Copyright John Maddock 2006
// Copyright Paul A. Bristow 2007, 2008, 2010

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifdef _MSC_VER
#  pragma warning(disable: 4512) // assignment operator could not be generated.
#  pragma warning(disable: 4510) // default constructor could not be generated.
#  pragma warning(disable: 4610) // can never be instantiated - user defined constructor required.
#  pragma warning(disable: 4180) // qualifier has no effect (in Fusion).
#endif

#include <iostream>
using std::cout; using std::endl;
using std::left; using std::fixed; using std::right; using std::scientific;
#include <iomanip>
using std::setw;
using std::setprecision;

#include <boost/math/distributions/fisher_f.hpp>

void f_test(
       double sd1,     // Sample 1 std deviation
       double sd2,     // Sample 2 std deviation
       double N1,      // Sample 1 size
       double N2,      // Sample 2 size
       double alpha)  // Significance level
{
   //
   // An F test applied to two sets of data.
   // We are testing the null hypothesis that the
   // standard deviation of the samples is equal, and
   // that any variation is down to chance.  We can
   // also test the alternative hypothesis that any
   // difference is not down to chance.
   // See http://www.itl.nist.gov/div898/handbook/eda/section3/eda359.htm
   //
   // Avoid "using namespace boost::math;" because of potential name ambiguity.
   using boost::math::fisher_f;

   // Print header:
   cout <<
      "____________________________________\n"
      "F test for equal standard deviations\n"
      "____________________________________\n\n";
   cout << setprecision(5);
   cout << "Sample 1:\n";
   cout << setw(55) << left << "Number of Observations" << "=  " << N1 << "\n";
   cout << setw(55) << left << "Sample Standard Deviation" << "=  " << sd1 << "\n\n";
   cout << "Sample 2:\n";
   cout << setw(55) << left << "Number of Observations" << "=  " << N2 << "\n";
   cout << setw(55) << left << "Sample Standard Deviation" << "=  " << sd2 << "\n\n";
   //
   // Now we can calculate and output some stats:
   //
   // F-statistic:
   double F = (sd1 / sd2);
   F *= F;
   cout << setw(55) << left << "Test Statistic" << "=  " << F << "\n\n";
   //
   // Finally define our distribution, and get the probability:
   //
   fisher_f dist(N1 - 1, N2 - 1);
   double p = cdf(dist, F);
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
   // Finally print out results of null and alternative hypothesis:
   //
   cout << setw(55) << left <<
      "Results for Alternative Hypothesis and alpha" << "=  "
      << setprecision(4) << fixed << alpha << "\n\n";
   cout << "Alternative Hypothesis                                    Conclusion\n";
   cout << "Standard deviations are unequal (two sided test)          ";
   if((ucv2 < F) || (lcv2 > F))
      cout << "NOT REJECTED\n";
   else
      cout << "REJECTED\n";
   cout << "Standard deviation 1 is less than standard deviation 2    ";
   if(lcv > F)
      cout << "NOT REJECTED\n";
   else
      cout << "REJECTED\n";
   cout << "Standard deviation 1 is greater than standard deviation 2 ";
   if(ucv < F)
      cout << "NOT REJECTED\n";
   else
      cout << "REJECTED\n";
   cout << endl << endl;
}

int main()
{
   //
   // Run tests for ceramic strength data:
   // see http://www.itl.nist.gov/div898/handbook/eda/section4/eda42a1.htm
   // The data for this case study were collected by Said Jahanmir of the
   // NIST Ceramics Division in 1996 in connection with a NIST/industry
   // ceramics consortium for strength optimization of ceramic strength.
   //
   f_test(65.54909, 61.85425, 240, 240, 0.05);
   //
   // And again for the process change comparison:
   // see http://www.itl.nist.gov/div898/handbook/prc/section3/prc32.htm
   // A new procedure to assemble a device is introduced and tested for
   // possible improvement in time of assembly. The question being addressed
   // is whether the standard deviation of the new assembly process (sample 2) is
   // better (i.e., smaller) than the standard deviation for the old assembly
   // process (sample 1).
   //
   f_test(4.9082, 2.5874, 11, 9, 0.05);
   return 0;
}

/*

Output:

  f_test.cpp
  F-test_example1.vcxproj -> J:\Cpp\MathToolkit\test\Math_test\Debug\F_test_example1.exe
  ____________________________________
  F test for equal standard deviations
  ____________________________________
  
  Sample 1:
  Number of Observations                                 =  240
  Sample Standard Deviation                              =  65.549
  
  Sample 2:
  Number of Observations                                 =  240
  Sample Standard Deviation                              =  61.854
  
  Test Statistic                                         =  1.123
  
  CDF of test statistic:                                 =  8.148e-001
  Upper Critical Value at alpha:                         =  1.238e+000
  Upper Critical Value at alpha/2:                       =  1.289e+000
  Lower Critical Value at alpha:                         =  8.080e-001
  Lower Critical Value at alpha/2:                       =  7.756e-001
  
  Results for Alternative Hypothesis and alpha           =  0.0500
  
  Alternative Hypothesis                                    Conclusion
  Standard deviations are unequal (two sided test)          REJECTED
  Standard deviation 1 is less than standard deviation 2    REJECTED
  Standard deviation 1 is greater than standard deviation 2 REJECTED
  
  
  ____________________________________
  F test for equal standard deviations
  ____________________________________
  
  Sample 1:
  Number of Observations                                 =  11.00000
  Sample Standard Deviation                              =  4.90820
  
  Sample 2:
  Number of Observations                                 =  9.00000
  Sample Standard Deviation                              =  2.58740
  
  Test Statistic                                         =  3.59847
  
  CDF of test statistic:                                 =  9.589e-001
  Upper Critical Value at alpha:                         =  3.347e+000
  Upper Critical Value at alpha/2:                       =  4.295e+000
  Lower Critical Value at alpha:                         =  3.256e-001
  Lower Critical Value at alpha/2:                       =  2.594e-001
  
  Results for Alternative Hypothesis and alpha           =  0.0500
  
  Alternative Hypothesis                                    Conclusion
  Standard deviations are unequal (two sided test)          REJECTED
  Standard deviation 1 is less than standard deviation 2    REJECTED
  Standard deviation 1 is greater than standard deviation 2 NOT REJECTED
  
  
  ____________________________________
  F test for equal standard deviations
  ____________________________________
  
  Sample 1:
  Number of Observations                                 =  240
  Sample Standard Deviation                              =  65.549
  
  Sample 2:
  Number of Observations                                 =  240
  Sample Standard Deviation                              =  61.854
  
  Test Statistic                                         =  1.123
  
  CDF of test statistic:                                 =  8.148e-001
  Upper Critical Value at alpha:                         =  1.238e+000
  Upper Critical Value at alpha/2:                       =  1.289e+000
  Lower Critical Value at alpha:                         =  8.080e-001
  Lower Critical Value at alpha/2:                       =  7.756e-001
  
  Results for Alternative Hypothesis and alpha           =  0.0500
  
  Alternative Hypothesis                                    Conclusion
  Standard deviations are unequal (two sided test)          REJECTED
  Standard deviation 1 is less than standard deviation 2    REJECTED
  Standard deviation 1 is greater than standard deviation 2 REJECTED
  
  
  ____________________________________
  F test for equal standard deviations
  ____________________________________
  
  Sample 1:
  Number of Observations                                 =  11.00000
  Sample Standard Deviation                              =  4.90820
  
  Sample 2:
  Number of Observations                                 =  9.00000
  Sample Standard Deviation                              =  2.58740
  
  Test Statistic                                         =  3.59847
  
  CDF of test statistic:                                 =  9.589e-001
  Upper Critical Value at alpha:                         =  3.347e+000
  Upper Critical Value at alpha/2:                       =  4.295e+000
  Lower Critical Value at alpha:                         =  3.256e-001
  Lower Critical Value at alpha/2:                       =  2.594e-001
  
  Results for Alternative Hypothesis and alpha           =  0.0500
  
  Alternative Hypothesis                                    Conclusion
  Standard deviations are unequal (two sided test)          REJECTED
  Standard deviation 1 is less than standard deviation 2    REJECTED
  Standard deviation 1 is greater than standard deviation 2 NOT REJECTED
  
 

*/

