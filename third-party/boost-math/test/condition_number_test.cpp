//  (C) Copyright Nick Thompson, 2019
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#include <cmath>
#include <limits>
#include "math_unit_test.hpp"
#include <boost/math/special_functions/lambert_w.hpp>
#include <boost/math/tools/condition_numbers.hpp>
#if !defined(TEST) || (TEST > 1)
#include <boost/multiprecision/cpp_bin_float.hpp>
#endif

using std::abs;
using std::log;
using boost::math::tools::summation_condition_number;
using boost::math::tools::evaluation_condition_number;

template<class Real>
void test_summation_condition_number()
{
    Real tol = 1000*std::numeric_limits<float>::epsilon();
    auto cond = summation_condition_number<Real>();
    // I've checked that the condition number increases with max_n,
    // and that the computed sum gets more accurate with increasing max_n.
    // But the CI system would die with more terms.
    Real max_n = 10000;
    for (Real n = 1; n < max_n; n += 2)
    {
        cond += 1/n;
        cond -= 1/(n+1);
    }

    CHECK_ABSOLUTE_ERROR(cond.sum(), log(Real(2)), tol);
    CHECK_GE(cond(), Real(14));
}

template<class Real>
void test_exponential_sum()
{
    using std::exp;
    using std::abs;
    Real eps = std::numeric_limits<float>::epsilon();
    for (Real x = -20; x <= -1; x += 0.5)
    {
        auto cond = summation_condition_number<Real>(1);
        size_t n = 1;
        Real term = x;
        while(n++ < 1000)
        {
            cond += term;
            term *= (x/n);
        }
        CHECK_ABSOLUTE_ERROR(exp(x), cond.sum(), eps*cond()*exp(x));
        CHECK_ABSOLUTE_ERROR(exp(2*abs(x)), cond(), eps*cond()*exp(2*abs(x)));
    }
}



template<class Real>
void test_evaluation_condition_number()
{
    using std::abs;
    using std::log;
    using std::sqrt;
    using std::exp;
    using std::sin;
    using std::tan;
    Real tol = sqrt(std::numeric_limits<Real>::epsilon());

    auto f1 = [](auto x) { return log(x); };
    for (Real x = 1.125; x < 8; x += 0.125)
    {
        Real cond = evaluation_condition_number(f1, x);
        CHECK_ABSOLUTE_ERROR(cond, 1/log(x), tol);
    }

    auto f2 = [](auto x) { return exp(x); };
    for (Real x = 1.125; x < 8; x += 0.125)
    {
        Real cond = evaluation_condition_number(f2, x);
        CHECK_ABSOLUTE_ERROR(cond, x, tol);
    }

    auto f3 = [](auto x) { return sin(x); };
    for (Real x = 1.125; x < 8; x += 0.125)
    {
        Real cond = evaluation_condition_number(f3, x);
        CHECK_ABSOLUTE_ERROR(cond, abs(x/tan(x)), tol);
    }

    // Test a function which right differentiable:
    using boost::math::constants::e;
    auto f4 = [](Real x) { return boost::math::lambert_w0(x); };
    Real cond = evaluation_condition_number(f4, -1/e<Real>());
    if (std::is_same<Real, float>::value)
    {
        CHECK_GE(cond, Real(30));
    }
    else
    {
        CHECK_GE(cond, Real(4900));
    }
}

int main()
{
#if !defined(TEST) || (TEST == 1)
    test_summation_condition_number<float>();
    test_evaluation_condition_number<float>();
    test_evaluation_condition_number<double>();
    test_evaluation_condition_number<long double>();
    test_exponential_sum<double>();
#endif
#if !defined(TEST) || (TEST == 2)
    test_summation_condition_number<boost::multiprecision::cpp_bin_float_50>();
#endif
#if !defined(TEST) || (TEST == 3)
    test_evaluation_condition_number<boost::multiprecision::cpp_bin_float_50>();
#endif
    return boost::math::test::report_errors();
}
