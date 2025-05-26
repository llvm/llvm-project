// Copyright Matt Borland, 2022
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "math_unit_test.hpp"
#include <boost/math/distributions/non_central_t.hpp>

template <typename T>
void test()
{
    boost::math::non_central_t_distribution<T> nct(T(10), T(3));

    // https://www.wolframalpha.com/input?i2d=true&i=N%5C%2891%29Kurtosis%5C%2891%29NoncentralStudentTDistribution%5C%2891%2910%5C%2844%29+3%5C%2893%29%5C%2893%29%5C%2844%2930%5C%2893%29
    CHECK_MOLLIFIED_CLOSE(boost::math::kurtosis(nct), T(5.44234533171835241424739188827L), 1e-6);

    boost::math::non_central_t_distribution<T> nct0(T(10), T(0));

    // https://www.wolframalpha.com/input?i2d=true&i=Kurtosis%5C%2891%29NoncentralStudentTDistribution%5C%2891%2910%5C%2844%29+0%5C%2893%29%5C%2893%29
    CHECK_ULP_CLOSE(boost::math::kurtosis(nct0), T(4), 1);

    // https://www.wolframalpha.com/input?i2d=true&i=ExcessKurtosis%5C%2891%29NoncentralStudentTDistribution%5C%2891%2910%5C%2844%29+0%5C%2893%29%5C%2893%29
    CHECK_ULP_CLOSE(boost::math::kurtosis_excess(nct0), T(1), 1);
}

int main(void)
{
    test<float>();
    test<double>();
    test<long double>();

    return boost::math::test::report_errors();
}
