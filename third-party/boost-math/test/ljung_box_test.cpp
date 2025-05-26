/*
 * Copyright Nick Thompson, 2019
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "math_unit_test.hpp"
#include <numeric>
#include <utility>
#include <random>
#include <boost/math/statistics/ljung_box.hpp>

using boost::math::statistics::ljung_box;

template<class Real>
void test_trivial()
{

  // Validate in R:
  // > v <- c(1,2)
  // > Box.test(v, lag=1, "Ljung")
  // Box-Ljung test
  // data:  v
  // X-squared = 2, df = 1, p-value = 0.1573

  std::vector<Real> v{1,2};

  double expected_statistic = 2;
  double expected_pvalue = 0.15729920705028455;

  auto [computed_statistic, computed_pvalue] = ljung_box(v, 1);
  CHECK_ULP_CLOSE(expected_statistic, computed_statistic, 10);
  CHECK_ULP_CLOSE(expected_pvalue, computed_pvalue, 30);
}


void test_agreement_with_mathematica()
{
  std::vector<double> v{0.7739928761039216,-0.4468259278452086,0.98287381303903,-0.3943029116201079,0.6569015496559457};
  double expected_statistic = 10.2076093223439;
  double expected_pvalue = 0.00607359458123835072;

  auto [computed_statistic, computed_pvalue] = ljung_box(v);
  CHECK_ULP_CLOSE(expected_statistic, computed_statistic, 3);
  CHECK_ULP_CLOSE(expected_pvalue, computed_pvalue, 3);
}

int main()
{
    test_trivial<double>();
    test_agreement_with_mathematica();
    return boost::math::test::report_errors();
}
