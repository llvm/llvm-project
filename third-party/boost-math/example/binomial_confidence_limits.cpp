// Copyright John Maddock 2006
// Copyright Paul A. Bristow 2010

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
#include <iomanip>
using std::fixed; using std::left; using std::right; using std::right; using std::setw;
using std::setprecision;

#include <boost/math/distributions/binomial.hpp>

void confidence_limits_on_frequency(unsigned trials, unsigned successes)
{
   //
   // trials = Total number of trials.
   // successes = Total number of observed successes.
   //
   // Calculate confidence limits for an observed
   // frequency of occurrence that follows a binomial distribution.
   //
   //using namespace std; // Avoid
   // using namespace boost::math; // potential name ambiguity with std <random>
   using boost::math::binomial_distribution;

   // Print out general info:
   cout <<
      "___________________________________________\n"
      "2-Sided Confidence Limits For Success Ratio\n"
      "___________________________________________\n\n";
   cout << setprecision(7);
   cout << setw(40) << left << "Number of Observations" << "=  " << trials << "\n";
   cout << setw(40) << left << "Number of successes" << "=  " << successes << "\n";
   cout << setw(40) << left << "Sample frequency of occurrence" << "=  " << double(successes) / trials << "\n";
   //
   // Define a table of significance levels:
   //
   double alpha[] = { 0.5, 0.25, 0.1, 0.05, 0.01, 0.001, 0.0001, 0.00001 };
   //
   // Print table header:
   //
   cout << "\n\n"
           "_______________________________________________________________________\n"
           "Confidence        Lower CP       Upper CP       Lower JP       Upper JP\n"
           " Value (%)        Limit          Limit          Limit          Limit\n"
           "_______________________________________________________________________\n";
   //
   // Now print out the data for the table rows.
   //
   for(unsigned i = 0; i < sizeof(alpha)/sizeof(alpha[0]); ++i)
   {
      // Confidence value:
      cout << fixed << setprecision(3) << setw(10) << right << 100 * (1-alpha[i]);
      // Calculate Clopper Pearson bounds:
      double l = binomial_distribution<>::find_lower_bound_on_p(trials, successes, alpha[i]/2);
      double u = binomial_distribution<>::find_upper_bound_on_p(trials, successes, alpha[i]/2);
      // Print Clopper Pearson Limits:
      cout << fixed << setprecision(5) << setw(15) << right << l;
      cout << fixed << setprecision(5) << setw(15) << right << u;
      // Calculate Jeffreys Prior Bounds:
      l = binomial_distribution<>::find_lower_bound_on_p(trials, successes, alpha[i]/2, binomial_distribution<>::jeffreys_prior_interval);
      u = binomial_distribution<>::find_upper_bound_on_p(trials, successes, alpha[i]/2, binomial_distribution<>::jeffreys_prior_interval);
      // Print Jeffreys Prior Limits:
      cout << fixed << setprecision(5) << setw(15) << right << l;
      cout << fixed << setprecision(5) << setw(15) << right << u << std::endl;
   }
   cout << endl;
} // void confidence_limits_on_frequency()

int main()
{
   confidence_limits_on_frequency(20, 4);
   confidence_limits_on_frequency(200, 40);
   confidence_limits_on_frequency(2000, 400);

   return 0;
} // int main()

/*

------ Build started: Project: binomial_confidence_limits, Configuration: Debug Win32 ------
Compiling...
binomial_confidence_limits.cpp
Linking...
Autorun "i:\boost-06-05-03-1300\libs\math\test\Math_test\debug\binomial_confidence_limits.exe"
___________________________________________
2-Sided Confidence Limits For Success Ratio
___________________________________________

Number of Observations                  =  20
Number of successes                     =  4
Sample frequency of occurrence          =  0.2


_______________________________________________________________________
Confidence        Lower CP       Upper CP       Lower JP       Upper JP
 Value (%)        Limit          Limit          Limit          Limit
_______________________________________________________________________
    50.000        0.12840        0.29588        0.14974        0.26916
    75.000        0.09775        0.34633        0.11653        0.31861
    90.000        0.07135        0.40103        0.08734        0.37274
    95.000        0.05733        0.43661        0.07152        0.40823
    99.000        0.03576        0.50661        0.04655        0.47859
    99.900        0.01905        0.58632        0.02634        0.55960
    99.990        0.01042        0.64997        0.01530        0.62495
    99.999        0.00577        0.70216        0.00901        0.67897

___________________________________________
2-Sided Confidence Limits For Success Ratio
___________________________________________

Number of Observations                  =  200
Number of successes                     =  40
Sample frequency of occurrence          =  0.2000000


_______________________________________________________________________
Confidence        Lower CP       Upper CP       Lower JP       Upper JP
 Value (%)        Limit          Limit          Limit          Limit
_______________________________________________________________________
    50.000        0.17949        0.22259        0.18190        0.22001
    75.000        0.16701        0.23693        0.16934        0.23429
    90.000        0.15455        0.25225        0.15681        0.24956
    95.000        0.14689        0.26223        0.14910        0.25951
    99.000        0.13257        0.28218        0.13468        0.27940
    99.900        0.11703        0.30601        0.11902        0.30318
    99.990        0.10489        0.32652        0.10677        0.32366
    99.999        0.09492        0.34485        0.09670        0.34197

___________________________________________
2-Sided Confidence Limits For Success Ratio
___________________________________________

Number of Observations                  =  2000
Number of successes                     =  400
Sample frequency of occurrence          =  0.2000000


_______________________________________________________________________
Confidence        Lower CP       Upper CP       Lower JP       Upper JP
 Value (%)        Limit          Limit          Limit          Limit
_______________________________________________________________________
    50.000        0.19382        0.20638        0.19406        0.20613
    75.000        0.18965        0.21072        0.18990        0.21047
    90.000        0.18537        0.21528        0.18561        0.21503
    95.000        0.18267        0.21821        0.18291        0.21796
    99.000        0.17745        0.22400        0.17769        0.22374
    99.900        0.17150        0.23079        0.17173        0.23053
    99.990        0.16658        0.23657        0.16681        0.23631
    99.999        0.16233        0.24169        0.16256        0.24143

*/



