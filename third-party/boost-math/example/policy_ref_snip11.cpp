//  Copyright John Maddock 2007.
//  Copyright Paul A. Bristow 2010
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Note that this file contains quickbook mark-up as well as code
// and comments, don't change any of the special comment mark-ups!

#include <iostream>
using std::cout; using std::endl;

// Setting (approximate) precision 25 bits in a single function call using make_policy.

//[policy_ref_snip11

#include <boost/math/distributions/normal.hpp>
using boost::math::normal_distribution;

using namespace boost::math::policies;

const int bits = 25; // approximate precision.

double q = quantile(
      normal_distribution<double, policy<digits2<bits> > >(), 
      0.05); // 5% quantile.

//] //[/policy_ref_snip11]

int main()
{
  std::streamsize p = 2 + (bits * 30103UL) / 100000UL; 
  // Approximate number of significant decimal digits for 25 bits.
  cout.precision(p); 
  cout << bits << " binary bits is approximately equivalent to " << p << " decimal digits " << endl;
  cout << "quantile(normal_distribution<double, policy<digits2<25> > >(), 0.05  = "
     << q << endl; // -1.64485
}

/*
Output:
  25 binary bits is approximately equivalent to 9 decimal digits 
  quantile(normal_distribution<double, policy<digits2<25> > >(), 0.05  = -1.64485363
  */

