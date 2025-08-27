// neg_binomial_sample_sizes.cpp

// Copyright John Maddock 2006
// Copyright Paul A. Bristow 2007, 2010

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/math/distributions/negative_binomial.hpp>
using boost::math::negative_binomial;

// Default RealType is double so this permits use of:
double find_minimum_number_of_trials(
double k,     // number of failures (events), k >= 0.
double p,     // fraction of trails for which event occurs, 0 <= p <= 1.
double probability); // probability threshold, 0 <= probability <= 1.

#include <iostream>
using std::cout;
using std::endl;
using std::fixed;
using std::right;
#include <iomanip>
using std::setprecision;
using std::setw; 

//[neg_binomial_sample_sizes

/*`
It centres around a routine that prints out a table of 
minimum sample sizes (number of trials) for various probability thresholds:
*/
  void find_number_of_trials(double failures, double p);
/*`
First define a table of significance levels: these are the maximum 
acceptable probability that /failure/ or fewer events will be observed.
*/
  double alpha[] = { 0.5, 0.25, 0.1, 0.05, 0.01, 0.001, 0.0001, 0.00001 };
/*`
Confidence value as % is (1 - alpha) * 100, so alpha 0.05 == 95% confidence
that the desired number of failures will be observed.
The values range from a very low 0.5 or 50% confidence up to an extremely high
confidence of 99.999.

Much of the rest of the program is pretty-printing, the important part
is in the calculation of minimum number of trials required for each
value of alpha using:

  (int)ceil(negative_binomial::find_minimum_number_of_trials(failures, p, alpha[i]);

find_minimum_number_of_trials returns a double,
so `ceil` rounds this up to ensure we have an integral minimum number of trials.
*/
  
void find_number_of_trials(double failures, double p)
{
   // trials = number of trials
   // failures = number of failures before achieving required success(es).
   // p        = success fraction (0 <= p <= 1.).
   //
   // Calculate how many trials we need to ensure the
   // required number of failures DOES exceed "failures".

  cout << "\n""Target number of failures = " << (int)failures;
  cout << ",   Success fraction = " << fixed << setprecision(1) << 100 * p << "%" << endl;
   // Print table header:
   cout << "____________________________\n"
           "Confidence        Min Number\n"
           " Value (%)        Of Trials \n"
           "____________________________\n";
   // Now print out the data for the alpha table values.
  for(unsigned i = 0; i < sizeof(alpha)/sizeof(alpha[0]); ++i)
   { // Confidence values %:
      cout << fixed << setprecision(3) << setw(10) << right << 100 * (1-alpha[i]) << "      "
      // find_minimum_number_of_trials
      << setw(6) << right
      << (int)ceil(negative_binomial::find_minimum_number_of_trials(failures, p, alpha[i]))
      << endl;
   }
   cout << endl;
} // void find_number_of_trials(double failures, double p)

/*` finally we can produce some tables of minimum trials for the chosen confidence levels:
*/

int main()
{
    find_number_of_trials(5, 0.5);
    find_number_of_trials(50, 0.5);
    find_number_of_trials(500, 0.5);
    find_number_of_trials(50, 0.1);
    find_number_of_trials(500, 0.1);
    find_number_of_trials(5, 0.9);

    return 0;
} // int main()

//]  [/neg_binomial_sample_sizes.cpp end of Quickbook in C++ markup]

/*

Output is:
Target number of failures = 5,   Success fraction = 50.0%
  ____________________________
  Confidence        Min Number
   Value (%)        Of Trials 
  ____________________________
      50.000          11
      75.000          14
      90.000          17
      95.000          18
      99.000          22
      99.900          27
      99.990          31
      99.999          36
  
  
  Target number of failures = 50,   Success fraction = 50.0%
  ____________________________
  Confidence        Min Number
   Value (%)        Of Trials 
  ____________________________
      50.000         101
      75.000         109
      90.000         115
      95.000         119
      99.000         128
      99.900         137
      99.990         146
      99.999         154
  
  
  Target number of failures = 500,   Success fraction = 50.0%
  ____________________________
  Confidence        Min Number
   Value (%)        Of Trials 
  ____________________________
      50.000        1001
      75.000        1023
      90.000        1043
      95.000        1055
      99.000        1078
      99.900        1104
      99.990        1126
      99.999        1146
  
  
  Target number of failures = 50,   Success fraction = 10.0%
  ____________________________
  Confidence        Min Number
   Value (%)        Of Trials 
  ____________________________
      50.000          56
      75.000          58
      90.000          60
      95.000          61
      99.000          63
      99.900          66
      99.990          68
      99.999          71
  
  
  Target number of failures = 500,   Success fraction = 10.0%
  ____________________________
  Confidence        Min Number
   Value (%)        Of Trials 
  ____________________________
      50.000         556
      75.000         562
      90.000         567
      95.000         570
      99.000         576
      99.900         583
      99.990         588
      99.999         594
  
  
  Target number of failures = 5,   Success fraction = 90.0%
  ____________________________
  Confidence        Min Number
   Value (%)        Of Trials 
  ____________________________
      50.000          57
      75.000          73
      90.000          91
      95.000         103
      99.000         127
      99.900         159
      99.990         189
      99.999         217
  
  
  Target number of failures = 5,   Success fraction = 95.0%
  ____________________________
  Confidence        Min Number
   Value (%)        Of Trials 
  ____________________________
      50.000         114
      75.000         148
      90.000         184
      95.000         208
      99.000         259
      99.900         324
      99.990         384
      99.999         442

*/
