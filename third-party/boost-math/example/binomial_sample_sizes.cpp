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

void find_max_sample_size(double p, unsigned successes)
{
   //
   // p         = success ratio.
   // successes = Total number of observed successes.
   //
   // Calculate how many trials we can have to ensure the
   // maximum number of successes does not exceed "successes".
   // A typical use would be failure analysis, where you want
   // zero or fewer "successes" with some probability.
   //
   // using namespace boost::math;
   // Avoid potential binomial_distribution name ambiguity with std <random>
   using boost::math::binomial_distribution;

   // Print out general info:
   cout <<
      "________________________\n"
      "Maximum Number of Trials\n"
      "________________________\n\n";
   cout << setprecision(7);
   cout << setw(40) << left << "Success ratio" << "=  " << p << "\n";
   cout << setw(40) << left << "Maximum Number of \"successes\" permitted" << "=  " << successes << "\n";
   //
   // Define a table of confidence intervals:
   //
   double alpha[] = { 0.5, 0.25, 0.1, 0.05, 0.01, 0.001, 0.0001, 0.00001 };
   //
   // Print table header:
   //
   cout << "\n\n"
           "____________________________\n"
           "Confidence        Max Number\n" 
           " Value (%)        Of Trials \n"
           "____________________________\n";
   //
   // Now print out the data for the table rows.
   //
   for(unsigned i = 0; i < sizeof(alpha)/sizeof(alpha[0]); ++i)
   {
      // Confidence value:
      cout << fixed << setprecision(3) << setw(10) << right << 100 * (1-alpha[i]);
      // calculate trials:
      double t = binomial_distribution<>::find_maximum_number_of_trials(successes, p, alpha[i]);
      t = floor(t);
      // Print Trials:
      cout << fixed << setprecision(0) << setw(15) << right << t << endl;
   }
   cout << endl;
}

int main()
{
   find_max_sample_size(1.0/1000, 0);
   find_max_sample_size(1.0/10000, 0);
   find_max_sample_size(1.0/100000, 0);
   find_max_sample_size(1.0/1000000, 0);

   return 0;
}


/*

Output:

  binomial_sample_sizes.cpp
  binomial_sample_sizes_example.vcxproj -> J:\Cpp\MathToolkit\test\Math_test\Debug\binomial_sample_sizes_example.exe
  ________________________
  Maximum Number of Trials
  ________________________
  
  Success ratio                           =  0.001
  Maximum Number of "successes" permitted =  0
  
  
  ____________________________
  Confidence        Max Number
   Value (%)        Of Trials 
  ____________________________
      50.000            692
      75.000            287
      90.000            105
      95.000             51
      99.000             10
      99.900              0
      99.990              0
      99.999              0
  
  ________________________
  Maximum Number of Trials
  ________________________
  
  Success ratio                           =  0.0001000
  Maximum Number of "successes" permitted =  0
  
  
  ____________________________
  Confidence        Max Number
   Value (%)        Of Trials 
  ____________________________
      50.000           6931
      75.000           2876
      90.000           1053
      95.000            512
      99.000            100
      99.900             10
      99.990              0
      99.999              0
  
  ________________________
  Maximum Number of Trials
  ________________________
  
  Success ratio                           =  0.0000100
  Maximum Number of "successes" permitted =  0
  
  
  ____________________________
  Confidence        Max Number
   Value (%)        Of Trials 
  ____________________________
      50.000          69314
      75.000          28768
      90.000          10535
      95.000           5129
      99.000           1005
      99.900            100
      99.990             10
      99.999              1
  
  ________________________
  Maximum Number of Trials
  ________________________
  
  Success ratio                           =  0.0000010
  Maximum Number of "successes" permitted =  0
  
  
  ____________________________
  Confidence        Max Number
   Value (%)        Of Trials 
  ____________________________
      50.000         693146
      75.000         287681
      90.000         105360
      95.000          51293
      99.000          10050
      99.900           1000
      99.990            100
      99.999             10
  

*/
