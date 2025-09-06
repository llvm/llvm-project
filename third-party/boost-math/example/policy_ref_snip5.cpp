//  Copyright John Maddock 2007.
//  Copyright Paul A. Bristow 2010.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Note that this file contains quickbook mark-up as well as code
// and comments, don't change any of the special comment mark-ups!

//[policy_ref_snip5

#include <boost/math/distributions/negative_binomial.hpp>
using boost::math::negative_binomial_distribution;

using namespace boost::math::policies;

typedef negative_binomial_distribution<
      double, 
      policy<discrete_quantile<real> > 
   > dist_type;
   
// Lower 5% quantile:
double x = quantile(dist_type(20, 0.3), 0.05);
// Upper 95% quantile:
double y = quantile(complement(dist_type(20, 0.3), 0.05));

//] //[/policy_ref_snip5]

#include <iostream>
using std::cout; using std::endl;

int main()
{
  cout << "quantile(dist_type(20, 0.3), 0.05)  = " << x 
    << "\nquantile(complement(dist_type(20, 0.3), 0.05) = " << y << endl;
}

/*

Output:
  quantile(dist_type(20, 0.3), 0.05)  = 27.3898
  quantile(complement(dist_type(20, 0.3), 0.05) = 68.1584

  */

