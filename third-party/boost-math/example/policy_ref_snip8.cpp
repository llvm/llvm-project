//  Copyright John Maddock 2007.
//  Copyright Paul A. Bristow 2010.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Note that this file contains quickbook mark-up as well as code
// and comments, don't change any of the special comment mark-ups!

//[policy_ref_snip8

#include <boost/math/distributions/negative_binomial.hpp>
using boost::math::negative_binomial_distribution;

using namespace boost::math::policies;

typedef negative_binomial_distribution<
      double, 
      policy<discrete_quantile<integer_round_nearest> > 
   > dist_type;
   
// Lower quantile rounded (down) to nearest:
double x = quantile(dist_type(20, 0.3), 0.05); // 27
// Upper quantile rounded (down) to nearest:
double y = quantile(complement(dist_type(20, 0.3), 0.05)); // 68

//] //[/policy_ref_snip8]

#include <iostream>
using std::cout; using std::endl;

int main()
{
   cout << "using policy<discrete_quantile<integer_round_nearest> " << endl
     << "quantile(dist_type(20, 0.3), 0.05) = " << x << endl
     << "quantile(complement(dist_type(20, 0.3), 0.05)) " << y << endl;
}

/*

Output:

   using policy<discrete_quantile<integer_round_nearest> 
  quantile(dist_type(20, 0.3), 0.05) = 27
  quantile(complement(dist_type(20, 0.3), 0.05)) 68

*/
