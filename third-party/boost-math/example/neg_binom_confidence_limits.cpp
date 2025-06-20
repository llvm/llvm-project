// neg_binomial_confidence_limits.cpp

// Copyright John Maddock 2006
// Copyright Paul A. Bristow 2007, 2010
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Caution: this file contains quickbook markup as well as code
// and comments, don't change any of the special comment markups!

//[neg_binomial_confidence_limits

/*`

First we need some includes to access the negative binomial distribution
(and some basic std output of course).

*/

#include <boost/math/distributions/negative_binomial.hpp>
using boost::math::negative_binomial;

#include <iostream>
using std::cout; using std::endl;
#include <iomanip>
using std::setprecision;
using std::setw; using std::left; using std::fixed; using std::right;

/*`
First define a table of significance levels: these are the 
probabilities that the true occurrence frequency lies outside the calculated
interval:
*/

  double alpha[] = { 0.5, 0.25, 0.1, 0.05, 0.01, 0.001, 0.0001, 0.00001 };

/*`
Confidence value as % is (1 - alpha) * 100, so alpha 0.05 == 95% confidence
that the true occurrence frequency lies *inside* the calculated interval.

We need a function to calculate and print confidence limits
for an observed frequency of occurrence 
that follows a negative binomial distribution.

*/

void confidence_limits_on_frequency(unsigned trials, unsigned successes)
{
   // trials = Total number of trials.
   // successes = Total number of observed successes.
   // failures = trials - successes.
   // success_fraction = successes /trials.
   // Print out general info:
   cout <<
      "______________________________________________\n"
      "2-Sided Confidence Limits For Success Fraction\n"
      "______________________________________________\n\n";
   cout << setprecision(7);
   cout << setw(40) << left << "Number of trials" << " =  " << trials << "\n";
   cout << setw(40) << left << "Number of successes" << " =  " << successes << "\n";
   cout << setw(40) << left << "Number of failures" << " =  " << trials - successes << "\n";
   cout << setw(40) << left << "Observed frequency of occurrence" << " =  " << double(successes) / trials << "\n";

   // Print table header:
   cout << "\n\n"
           "___________________________________________\n"
           "Confidence        Lower          Upper\n"
           " Value (%)        Limit          Limit\n"
           "___________________________________________\n";


/*`
And now for the important part - the bounds themselves.
For each value of /alpha/, we call `find_lower_bound_on_p` and 
`find_upper_bound_on_p` to obtain lower and upper bounds respectively. 
Note that since we are calculating a two-sided interval,
we must divide the value of alpha in two.  Had we been calculating a 
single-sided interval, for example: ['"Calculate a lower bound so that we are P%
sure that the true occurrence frequency is greater than some value"]
then we would *not* have divided by two.
*/

   // Now print out the upper and lower limits for the alpha table values.
   for(unsigned i = 0; i < sizeof(alpha)/sizeof(alpha[0]); ++i)
   {
      // Confidence value:
      cout << fixed << setprecision(3) << setw(10) << right << 100 * (1-alpha[i]);
      // Calculate bounds:
      double lower = negative_binomial::find_lower_bound_on_p(trials, successes, alpha[i]/2);
      double upper = negative_binomial::find_upper_bound_on_p(trials, successes, alpha[i]/2);
      // Print limits:
      cout << fixed << setprecision(5) << setw(15) << right << lower;
      cout << fixed << setprecision(5) << setw(15) << right << upper << endl;
   }
   cout << endl;
} // void confidence_limits_on_frequency(unsigned trials, unsigned successes)

/*`

And then call confidence_limits_on_frequency with increasing numbers of trials,
but always the same success fraction 0.1, or 1 in 10.

*/

int main()
{
  confidence_limits_on_frequency(20, 2); // 20 trials, 2 successes, 2 in 20, = 1 in 10 = 0.1 success fraction.
  confidence_limits_on_frequency(200, 20); // More trials, but same 0.1 success fraction.
  confidence_limits_on_frequency(2000, 200); // Many more trials, but same 0.1 success fraction.

  return 0;
} // int main()

//] [/negative_binomial_confidence_limits_eg end of Quickbook in C++ markup]

/*

______________________________________________
2-Sided Confidence Limits For Success Fraction
______________________________________________
Number of trials                         =  20
Number of successes                      =  2
Number of failures                       =  18
Observed frequency of occurrence         =  0.1
___________________________________________
Confidence        Lower          Upper
 Value (%)        Limit          Limit
___________________________________________
    50.000        0.04812        0.13554
    75.000        0.03078        0.17727
    90.000        0.01807        0.22637
    95.000        0.01235        0.26028
    99.000        0.00530        0.33111
    99.900        0.00164        0.41802
    99.990        0.00051        0.49202
    99.999        0.00016        0.55574
______________________________________________
2-Sided Confidence Limits For Success Fraction
______________________________________________
Number of trials                         =  200
Number of successes                      =  20
Number of failures                       =  180
Observed frequency of occurrence         =  0.1000000
___________________________________________
Confidence        Lower          Upper
 Value (%)        Limit          Limit
___________________________________________
    50.000        0.08462        0.11350
    75.000        0.07580        0.12469
    90.000        0.06726        0.13695
    95.000        0.06216        0.14508
    99.000        0.05293        0.16170
    99.900        0.04343        0.18212
    99.990        0.03641        0.20017
    99.999        0.03095        0.21664
______________________________________________
2-Sided Confidence Limits For Success Fraction
______________________________________________
Number of trials                         =  2000
Number of successes                      =  200
Number of failures                       =  1800
Observed frequency of occurrence         =  0.1000000
___________________________________________
Confidence        Lower          Upper
 Value (%)        Limit          Limit
___________________________________________
    50.000        0.09536        0.10445
    75.000        0.09228        0.10776
    90.000        0.08916        0.11125
    95.000        0.08720        0.11352
    99.000        0.08344        0.11802
    99.900        0.07921        0.12336
    99.990        0.07577        0.12795
    99.999        0.07282        0.13206
*/

