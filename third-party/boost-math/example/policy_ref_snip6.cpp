//  Copyright John Maddock 2007.
//  Copyright Paul A. Bristow 2010.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Note that this file contains quickbook mark-up as well as code
// and comments, don't change any of the special comment mark-ups!

//[policy_ref_snip6

#include <boost/math/distributions/negative_binomial.hpp>
using boost::math::negative_binomial;

// Use the default rounding policy integer_round_outwards.
// Lower quantile rounded down:
double x = quantile(negative_binomial(20, 0.3), 0.05); // rounded up 27 from 27.3898
// Upper quantile rounded up:
double y = quantile(complement(negative_binomial(20, 0.3), 0.05)); // rounded down to 69 from 68.1584

//] //[/policy_ref_snip6]

#include <iostream>
using std::cout; using std::endl;

int main()
{
   cout << "quantile(negative_binomial(20, 0.3), 0.05) = "<< x <<endl
     << "quantile(complement(negative_binomial(20, 0.3), 0.05)) = " << y << endl;
}

/*
Output:

  quantile(negative_binomial(20, 0.3), 0.05) = 27
  quantile(complement(negative_binomial(20, 0.3), 0.05)) = 69
*/

