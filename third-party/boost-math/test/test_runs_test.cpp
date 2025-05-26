/*
 * Copyright Nick Thompson, 2019
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "math_unit_test.hpp"
#include <vector>
#include <random>
#include <boost/math/statistics/runs_test.hpp>

using boost::math::statistics::runs_above_and_below_median;

void test_agreement_with_r_randtests()
{
  // $R
  // install.packages("randtests")
  // library(randtests)
  // earthden <- c(5.36, 5.29, 5.58, 5.65, 5.57, 5.53, 5.62, 5.29, 5.44, 5.34, 5.79,5.10, 5.27, 5.39, 5.42, 5.47, 5.63, 5.34, 5.46, 5.30, 5.75, 5.68, 5.85)
  // h = runs.test(earthden)
  // options(digits=18)
  //> h$statistic
  // -1.74772579501060576
  // > h$p.value
  // [1] 0.0805115199405023046
  // median of v is 5.46, 23 elements.
  std::vector<double> v{5.36, 5.29,
                        5.58, 5.65, 5.57, 5.53, 5.62,
                        5.29, 5.44, 5.34,
                        5.79, 5.10,
                        5.27, 5.39, 5.42,
                        5.47, 5.63,
                        5.34,
                        5.46, /* median */
                        5.30,
                        5.75, 5.68, 5.85};
  // v -> {-,-,+,+,+,+,+,-,-,-,+,+,-,-,-,+,+,-,-,+,+,+}, 8 runs.
  double expected_statistic = -1.74772579501060576;
  double expected_pvalue = 0.0805115199405023046;

  auto [computed_statistic, computed_pvalue] = runs_above_and_below_median(v);

  CHECK_ULP_CLOSE(expected_statistic, computed_statistic, 3);
  CHECK_ULP_CLOSE(expected_pvalue, computed_pvalue, 3);
}

void test_doc_example()
{
    std::vector<double> v{5, 2, 0, 4, 7, 9, 10, 6, 1, 8, 3};
    double expected_statistic = -0.670820393249936919;
    double expected_pvalue = 0.502334954360502017;

    auto [computed_statistic, computed_pvalue] = runs_above_and_below_median(v);

    CHECK_ULP_CLOSE(expected_statistic, computed_statistic, 3);
    CHECK_ULP_CLOSE(expected_pvalue, computed_pvalue, 3);
}

void test_constant_vector()
{
    std::vector<double> v{5,5,5,5,5,5,5};
    auto [computed_statistic, computed_pvalue] = runs_above_and_below_median(v);
    double expected_pvalue = 0;
    CHECK_ULP_CLOSE(expected_pvalue, computed_pvalue, 3);
    if (!std::isnan(computed_statistic)) {
        std::cerr << "Computed statistic is not a nan!\n";
    }
}

int main()
{
    test_constant_vector();
    test_agreement_with_r_randtests();
    test_doc_example();
    return boost::math::test::report_errors();
}
