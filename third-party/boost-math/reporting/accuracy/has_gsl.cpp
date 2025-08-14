//  Copyright John Maddock 2015.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <gsl/gsl_sf.h>

int main()
{
   double d = gsl_sf_bessel_Jn(2, 1.0);
   return d != 0 ? 0 : 1;
}
