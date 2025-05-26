//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "math_unit_test.hpp"
#include <boost/math/statistics/z_test.hpp>
#include <boost/math/statistics/univariate_statistics.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <limits>
#include <vector>
#include <random>
#include <utility>

using quad = boost::multiprecision::cpp_bin_float_quad;
using std::sqrt;

template<typename Real>
void test_one_sample_z()
{
    std::pair<Real, Real> temp = boost::math::statistics::one_sample_z_test(Real(10), Real(2), Real(100), Real(10));
    Real computed_statistic = std::get<0>(temp);
    Real computed_pvalue = std::get<1>(temp);
    CHECK_ULP_CLOSE(Real(0), computed_statistic, 5);
    CHECK_MOLLIFIED_CLOSE(Real(0), computed_pvalue, 5*std::numeric_limits<Real>::epsilon());

    temp = boost::math::statistics::one_sample_z_test(Real(10), Real(2), Real(100), Real(5));
    Real computed_statistic_2 = std::get<0>(temp);
    CHECK_ULP_CLOSE(Real(25), computed_statistic_2, 5);

    temp = boost::math::statistics::one_sample_z_test(Real(1)/2, Real(10), Real(100), Real(1)/3);
    Real computed_statistic_3 = std::get<0>(temp);
    CHECK_ULP_CLOSE(Real(1)/6, computed_statistic_3, 5);
}

template<typename Z>
void test_integer_one_sample_z()
{
    std::pair<double, double> temp = boost::math::statistics::one_sample_z_test(Z(10), Z(2), Z(100), Z(10));
    double computed_statistic = std::get<0>(temp);
    double computed_pvalue = std::get<1>(temp);
    CHECK_ULP_CLOSE(0.0, computed_statistic, 5);
    CHECK_MOLLIFIED_CLOSE(0.0, computed_pvalue, 5*std::numeric_limits<double>::epsilon());

    temp = boost::math::statistics::one_sample_z_test(Z(10), Z(2), Z(100), Z(5));
    double computed_statistic_2 = std::get<0>(temp);
    CHECK_ULP_CLOSE(25.0, computed_statistic_2, 5);
}

template<typename Real>
void test_two_sample_z()
{
    std::vector<Real> set_1 {1,2,3,4,5};
    std::vector<Real> set_2 {2,3,4,5,6};

    std::pair<Real, Real> temp = boost::math::statistics::two_sample_z_test(set_2, set_1);
    Real computed_statistic = std::get<0>(temp);
    Real computed_pvalue = std::get<1>(temp);
    CHECK_ULP_CLOSE(Real(1), computed_statistic, 5);
    CHECK_MOLLIFIED_CLOSE(Real(0), computed_pvalue, sqrt(std::numeric_limits<Real>::epsilon()));
}

template<typename Z>
void test_integer_two_sample_z()
{
    std::vector<Z> set_1 {1,2,3,4,5};
    std::vector<Z> set_2 {2,3,4,5,6};

    std::pair<double, double> temp = boost::math::statistics::two_sample_z_test(set_2, set_1);
    double computed_statistic = std::get<0>(temp);
    double computed_pvalue = std::get<1>(temp);
    CHECK_ULP_CLOSE(1.0, computed_statistic, 5);
    CHECK_MOLLIFIED_CLOSE(0.0, computed_pvalue, 5*std::numeric_limits<double>::epsilon());
}

int main()
{
    test_one_sample_z<float>();
    test_one_sample_z<double>();
    test_one_sample_z<quad>();

    test_integer_one_sample_z<int>();
    test_integer_one_sample_z<int32_t>();
    test_integer_one_sample_z<int64_t>();
    test_integer_one_sample_z<uint32_t>();

    test_two_sample_z<float>();
    test_two_sample_z<double>();
    test_two_sample_z<quad>();

    test_integer_two_sample_z<int>();
    test_integer_two_sample_z<int32_t>();
    test_integer_two_sample_z<int64_t>();
    test_integer_two_sample_z<uint32_t>();
}
