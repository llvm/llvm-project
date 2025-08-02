//  Copyright John Maddock 2024.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//
// The purpose of this test case is to probe the skew normal quantiles
// for extreme values of skewness and ensure that our root finders don't
// blow up, see https://github.com/boostorg/math/issues/1120 for original
// test case.  We test both the maximum number of iterations taken, and the
// overall total (ie average).  Any changes to the skew normal should
// ideally NOT cause this test to fail, as this indicates that our root
// finding has been made worse by the change!!
//
// Note that defining BOOST_MATH_INSTRUMENT_SKEW_NORMAL_ITERATIONS
// causes the skew normal quantile to save the number of iterations
// to a global variable "global_iter_count".
//

#define BOOST_MATH_INSTRUMENT_SKEW_NORMAL_ITERATIONS

#include <random>
#include <boost/math/distributions/skew_normal.hpp>
#include "math_unit_test.hpp"

std::uintmax_t global_iter_count;
std::uintmax_t total_iter_count = 0;

int main()
{
   using scipy_policy = boost::math::policies::policy<boost::math::policies::promote_double<false>>;

   std::mt19937 gen;
   std::uniform_real_distribution<double> location(-3, 3);
   std::uniform_real_distribution<double> scale(0.001, 3);

   for (unsigned skew = 50; skew < 2000; skew += 43)
   {
      constexpr double pn[] = { 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4 };
      boost::math::skew_normal_distribution<double, scipy_policy> dist(location(gen), scale(gen), skew);
      for (unsigned i = 0; i < 7; ++i)
      {
         global_iter_count = 0;
         double x = quantile(dist, pn[i]);
         total_iter_count += global_iter_count;
         CHECK_LE(global_iter_count, static_cast<std::uintmax_t>(60));
         double p = cdf(dist, x);
         CHECK_ABSOLUTE_ERROR(p, pn[i], 45 * std::numeric_limits<double>::epsilon());

         global_iter_count = 0;
         x = quantile(complement(dist, pn[i]));
         total_iter_count += global_iter_count;
         CHECK_LE(global_iter_count, static_cast<std::uintmax_t>(60));
         p = cdf(complement(dist, x));
         CHECK_ABSOLUTE_ERROR(p, pn[i], 45 * std::numeric_limits<double>::epsilon());
      }
   }
   CHECK_LE(total_iter_count, static_cast<std::uintmax_t>(10000));
   return boost::math::test::report_errors();
}
