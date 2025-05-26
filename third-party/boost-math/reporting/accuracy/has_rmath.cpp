//  Copyright John Maddock 2015.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define MATHLIB_STANDALONE
#include <Rmath.h>

int main()
{
   double d = psigamma(2.0, 4);
   return d != 0 ? 0 : 1;
}
