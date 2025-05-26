//  Copyright John Maddock 2015.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <math.h>

int main()
{
   long double d = 1;

   d = ::erf(d);
   d = ::erfc(d);
   d = ::tgamma(d);
   d = ::lgamma(d);
   return d != 0 ? 0 : 1;
}
