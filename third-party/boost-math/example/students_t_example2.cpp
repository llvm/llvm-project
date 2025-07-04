// students_t_example2.cpp

// Copyright Paul A. Bristow 2006.
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Example 2 of using Student's t

// A general guide to Student's t is at
// http://en.wikipedia.org/wiki/Student's_t-test
// (and many other elementary and advanced statistics texts).
// It says:
// The t statistic was invented by William Sealy Gosset
// for cheaply monitoring the quality of beer brews.
// "Student" was his pen name.
// Gosset was statistician for Guinness brewery in Dublin, Ireland,
// hired due to Claude Guinness's innovative policy of recruiting the
// best graduates from Oxford and Cambridge for applying biochemistry
// and statistics to Guinness's industrial processes.
// Gosset published the t test in Biometrika in 1908,
// but was forced to use a pen name by his employer who regarded the fact
// that they were using statistics as a trade secret.
// In fact, Gosset's identity was unknown not only to fellow statisticians
// but to his employer - the company insisted on the pseudonym
// so that it could turn a blind eye to the breach of its rules.

// The Students't distribution function is described at
// http://en.wikipedia.org/wiki/Student%27s_t_distribution

#include <boost/math/distributions/students_t.hpp>
   using boost::math::students_t;  // Probability of students_t(df, t).

#include <iostream>
   using std::cout;
   using std::endl;
#include <iomanip>
   using std::setprecision;
   using std::setw;
#include <cmath>
   using std::sqrt;

// This example of a one-sided test is from:
//
// from Statistics for Analytical Chemistry, 3rd ed. (1994), pp 59-60
// J. C. Miller and J. N. Miller, Ellis Horwood ISBN 0 13 0309907.

// An acid-base titrimetric method has a significant indicator error and
// thus tends to give results with a positive systematic error (+bias).
// To test this an exactly 0.1 M solution of acid is used to titrate
// 25.00 ml of exactly 0.1 M solution of alkali,
// with the following results (ml):

double reference = 25.00; // 'True' result.
const int values = 6; // titrations.
double data [values] = {25.06, 25.18, 24.87, 25.51, 25.34, 25.41};

int main()
{
   cout << "Example2 using Student's t function. ";
#if defined(__FILE__) && defined(__TIMESTAMP__) && defined(_MSC_FULL_VER)
   cout << "  " << __FILE__ << ' ' << __TIMESTAMP__ << ' '<< _MSC_FULL_VER;
#endif
   cout << endl;

   double sum = 0.;
   for (int value = 0; value < values; value++)
   { // Echo data and calculate mean.
      sum += data[value];
      cout << setw(4) << value << ' ' << setw(14) << data[value] << endl;
   }
   double mean = sum /static_cast<double>(values);
   cout << "Mean = " << mean << endl; // 25.2283

   double sd = 0.;
   for (int value = 0; value < values; value++)
   { // Calculate standard deviation.
      sd +=(data[value] - mean) * (data[value] - mean);
   }
   int degrees_of_freedom = values - 1; // Use the n-1 formula.
   sd /= degrees_of_freedom; // == variance.
   sd= sqrt(sd);
   cout << "Standard deviation = " << sd<< endl; // = 0.238279

   double t = (mean - reference) * sqrt(static_cast<double>(values))/ sd; //
   cout << "Student's t = " << t << ", with " << degrees_of_freedom << " degrees of freedom." << endl; // = 2.34725

   cout << "Probability of positive bias is " << cdf(students_t(degrees_of_freedom), t) << "."<< endl; // =  0.967108.
   // A 1-sided test because only testing for a positive bias.
   // If > 0.95 then greater than 1 in 20 conventional (arbitrary) requirement.

   return 0;
}  // int main()

/*

Output is:

------ Build started: Project: students_t_example2, Configuration: Debug Win32 ------
Compiling...
students_t_example2.cpp
Linking...
Autorun "i:\boost-06-05-03-1300\libs\math\test\Math_test\debug\students_t_example2.exe"
Example2 using Student's t function.   ..\..\..\..\..\..\boost-sandbox\libs\math_functions\example\students_t_example2.cpp Sat Aug 12 16:55:59 2006 140050727
   0          25.06
   1          25.18
   2          24.87
   3          25.51
   4          25.34
   5          25.41
Mean = 25.2283
Standard deviation = 0.238279
Student's t = 2.34725, with 5 degrees of freedom.
Probability of positive bias is 0.967108.
Build Time 0:03
Build log was saved at "file://i:\boost-06-05-03-1300\libs\math\test\Math_test\students_t_example2\Debug\BuildLog.htm"
students_t_example2 - 0 error(s), 0 warning(s)
========== Build: 1 succeeded, 0 failed, 0 up-to-date, 0 skipped ==========

*/





