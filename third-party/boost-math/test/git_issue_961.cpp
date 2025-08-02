//  (C) Copyright John Maddock 2023.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <iomanip>
#include <cfenv>
#include <boost/math/special_functions/beta.hpp>

using namespace std;
using namespace boost::math;

void show_fp_exception_flags()
{
   if (std::fetestexcept(FE_DIVBYZERO)) {
      cout << " FE_DIVBYZERO";
   }
   // FE_INEXACT is common and not interesting.
   // if (std::fetestexcept(FE_INEXACT)) {
   //     cout << " FE_INEXACT";
   // }
   if (std::fetestexcept(FE_INVALID)) {
      cout << " FE_INVALID";
   }
   if (std::fetestexcept(FE_OVERFLOW)) {
      cout << " FE_OVERFLOW";
   }
   if (std::fetestexcept(FE_UNDERFLOW)) {
      cout << " FE_UNDERFLOW";
   }
   cout << endl;
}

template <class Policy>
int test()
{
   double a = 14.208308325339239;
   double b = a;

   double p = 6.4898872103239473e-300;  // Throws exception: Assertion `x >= 0' failed.
   // double p = 7.8e-307;  // No flags set, returns 8.57354094063444939e-23
   // double p = 7.7e-307;  // FE_UNDERFLOW set, returns 0.0

   while (p > (std::numeric_limits<double>::min)())
   {
      std::feclearexcept(FE_ALL_EXCEPT);

      try {

         double x = ibeta_inv(a, b, p, Policy());

         show_fp_exception_flags();

         std::cout << std::scientific << std::setw(24)
            << std::setprecision(17) << x << std::endl;
      }
      catch (const std::exception& e)
      {
         std::cout << e.what() << std::endl;
         return 1;
      }

      p /= 1.25;
   }

   return 0;
}

int main(int argc, char* argv[])
{
   using namespace boost::math::policies;
   if (test<policy<>>())
      return 1;
   return test<policy<promote_double<false>>>();
}
