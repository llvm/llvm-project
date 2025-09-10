//  (C) Copyright Matt Borland 2023.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "math_unit_test.hpp"
#include <cfenv>
#include <iostream>
#include <boost/math/distributions/beta.hpp>

#pragma STDC FENV_ACCESS ON

// Show and then clear the fenv flags
void show_fpexcept_flags()
{
   bool fe = false;

   if (std::fetestexcept(FE_OVERFLOW))
   {
      fe = true;
      std::cerr << "FE_OVERFLOW" << std::endl;
   }
   if (std::fetestexcept(FE_UNDERFLOW))
   {
      //fe = true;
      std::cerr << "FE_UNDERFLOW" << std::endl;
   }
   if (std::fetestexcept(FE_DIVBYZERO))
   {
      fe = true;
      std::cerr << "FE_DIVBYZERO" << std::endl;
   }
   if (std::fetestexcept(FE_INVALID))
   {
      fe = true;
      std::cerr << "FE_INVALID" << std::endl;
   }

   CHECK_EQUAL(fe, false);

   std::feclearexcept(FE_ALL_EXCEPT);
}

int main()
{
   // Default Scipy policy
   using my_policy = boost::math::policies::policy<boost::math::policies::promote_float<false>, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy, boost::math::policies::default_policy>;

   double a = 1e-307;

   while (a)
   {
      const auto dist_a = boost::math::beta_distribution<double, my_policy>(1e-308, 5.0);
      const auto dist_a_ppf = boost::math::quantile(dist_a, 0.2);
      show_fpexcept_flags();
      CHECK_MOLLIFIED_CLOSE(dist_a_ppf, 0.0, 1e-10);
      a /= 2;
   }
   return boost::math::test::report_errors();
}
