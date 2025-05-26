//  (C) Copyright John Maddock 2005.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef NTL_STD_CXX
#  define NTL_STD_CXX
#endif

#include <iostream>
#include <iomanip>
#include "mp_t.hpp"

mp_t log1p(mp_t arg)
{
   return log(arg + 1);
}

mp_t expm1(mp_t arg)
{
   return exp(arg) - 1;
}

int main()
{
   mp_t r, root_two;
   r = 1.0;
   root_two = 2.0;
   root_two = sqrt(root_two);
   r /= root_two;
   mp_t lim = pow(mp_t(2), mp_t(-128));
   std::cout << std::scientific << std::setprecision(40);
   while(r > lim)
   {
      std::cout << "   { " << r << "L, " << log1p(r) << "L, " << expm1(r) << "L, }, \n";
      std::cout << "   { " << -r << "L, " << log1p(-r) << "L, " << expm1(-r) << "L, }, \n";
      r /= root_two;
   }
   return 0;
}

