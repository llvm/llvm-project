//  Copyright John Maddock 2015.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cmath>

int main()
{
   long double d = 1;

   d = std::erf(d);
   d = std::erfc(d);
   d = std::tgamma(d);
   d = std::lgamma(d);
   d = std::comp_ellint_1(d);
   return d != 0 ? 0 : 1;
}
