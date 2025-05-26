//  Copyright John Maddock 2015.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <tr1/cmath>

int main()
{
   long double d = 1;

   d = std::tr1::erf(d);
   d = std::tr1::erfc(d);
   d = std::tr1::tgamma(d);
   d = std::tr1::lgamma(d);
   return d != 0 ? 0 : 1;
}
