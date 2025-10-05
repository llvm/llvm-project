/*
 * Copyright Nick Thompson, 2021
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <boost/math/tools/config.hpp>
#ifndef BOOST_MATH_NO_THREAD_LOCAL_WITH_NON_TRIVIAL_TYPES
#include "math_unit_test.hpp"
#include <numeric>
#include <random>
#include <array>
#include <boost/core/demangle.hpp>
#include <boost/math/interpolators/bezier_polynomial.hpp>
#ifdef BOOST_HAS_FLOAT128
#include <boost/multiprecision/float128.hpp>
using boost::multiprecision::float128;
#endif

#if __has_include(<stdfloat>)
#  include <stdfloat>
#endif

using boost::math::interpolators::bezier_polynomial;

template<typename Real>
void test_linear()
{
    std::vector<std::array<Real, 2>> control_points(2);
    control_points[0] = {Real(0), Real(0)};
    control_points[1] = {Real(1), Real(1)};
    auto control_points_copy = control_points;
    auto bp = bezier_polynomial(std::move(control_points_copy));

    // P(0) = P_0:
    CHECK_ULP_CLOSE(control_points[0][0], bp(0)[0], 3);
    CHECK_ULP_CLOSE(control_points[0][1], bp(0)[1], 3);

    // P(1) = P_n:
    CHECK_ULP_CLOSE(control_points[1][0], bp(1)[0], 3);
    CHECK_ULP_CLOSE(control_points[1][1], bp(1)[1], 3);

    for (Real t = Real(1)/32; t < 1; t += Real(1)/32) {
        Real expected0 = (1-t)*control_points[0][0] + t*control_points[1][0];
        CHECK_ULP_CLOSE(expected0, bp(t)[0], 3);
    }

    // P(1) = P_n:
    std::array<Real, 2> endpoint{1,2};
    bp.edit_control_point(endpoint, 1);
    CHECK_ULP_CLOSE(endpoint[0], bp(1)[0], 3);
    CHECK_ULP_CLOSE(endpoint[1], bp(1)[1], 3);

}

template<typename Real>
void test_quadratic()
{
    std::vector<std::array<Real, 2>> control_points(3);
    control_points[0] = {Real(0), Real(0)};
    control_points[1] = {Real(1), Real(1)};
    control_points[2] = {Real(2), Real(2)};
    auto control_points_copy = control_points;
    auto bp = bezier_polynomial(std::move(control_points_copy));

    // P(0) = P_0:
    auto computed_point = bp(0);
    CHECK_ULP_CLOSE(control_points[0][0], computed_point[0], 3);
    CHECK_ULP_CLOSE(control_points[0][1], computed_point[1], 3);
    auto computed_dp = bp.prime(0);
    CHECK_ULP_CLOSE(2*(control_points[1][0] - control_points[0][0]), computed_dp[0], 3);
    CHECK_ULP_CLOSE(2*(control_points[1][1] - control_points[0][1]), computed_dp[1], 3);

    // P(1) = P_n:
    computed_point = bp(1);
    CHECK_ULP_CLOSE(control_points[2][0], computed_point[0], 3);
    CHECK_ULP_CLOSE(control_points[2][1], computed_point[1], 3);
}

// All points on a Bezier polynomial fall into the convex hull of the control polygon.
template<typename Real>
void test_convex_hull()
{
    std::vector<std::array<Real, 2>> control_points(4);
    control_points[0] = {Real(0), Real(0)};
    control_points[1] = {Real(0), Real(1)};
    control_points[2] = {Real(1), Real(1)};
    control_points[3] = {Real(1), Real(0)};
    auto bp = bezier_polynomial(std::move(control_points));

    for (Real t = 0; t <= 1; t += Real(1)/32) {
        auto p = bp(t);
        CHECK_LE(p[0], Real(1));
        CHECK_LE(Real(0), p[0]);
        CHECK_LE(p[1], Real(1));
        CHECK_LE(Real(0), p[1]);
    }
}

// Reversal Symmetry: If q(t) is the Bezier polynomial which consumes the control points in reversed order from p(t),
// then p(t) = q(1-t).
template<typename Real>
void test_reversal_symmetry()
{
    std::vector<std::array<Real, 3>> control_points(10);
    std::uniform_real_distribution<Real> dis(-1,1);
    std::mt19937_64 gen;
    for (size_t i = 0; i < control_points.size(); ++i) {
        for (size_t j = 0; j < 3; ++j) {
            control_points[i][j] = dis(gen);
        }
    }

    auto control_points_copy = control_points;
    auto bp0 = bezier_polynomial(std::move(control_points_copy));

    control_points_copy = control_points;
    std::reverse(control_points_copy.begin(), control_points_copy.end());
    auto bp1 = bezier_polynomial(std::move(control_points_copy));
    auto P0 = bp0(Real(0));
    CHECK_ULP_CLOSE(control_points[0][0], P0[0], 3);
    CHECK_ULP_CLOSE(control_points[0][1], P0[1], 3);
    CHECK_ULP_CLOSE(control_points[0][2], P0[2], 3);
    auto P1 = bp0(Real(1));
    CHECK_ULP_CLOSE(control_points.back()[0], P1[0], 3);
    CHECK_ULP_CLOSE(control_points.back()[1], P1[1], 3);
    CHECK_ULP_CLOSE(control_points.back()[2], P1[2], 3);

    P0 = bp1(Real(1));
    CHECK_ULP_CLOSE(control_points[0][0], P0[0], 3);
    CHECK_ULP_CLOSE(control_points[0][1], P0[1], 3);
    CHECK_ULP_CLOSE(control_points[0][2], P0[2], 3);

    P1 = bp1(Real(0));
    CHECK_ULP_CLOSE(control_points.back()[0], P1[0], 3);
    CHECK_ULP_CLOSE(control_points.back()[1], P1[1], 3);
    CHECK_ULP_CLOSE(control_points.back()[2], P1[2], 3);

    for (Real t = 0; t <= 1; t += Real(1.0)) {
        auto P0 = bp0(t);
        auto P1 = bp1(Real(1.0)-t);
        if (!CHECK_ULP_CLOSE(P0[0], P1[0], 3)) {
            std::cerr << "  Error at t = " << t << "\n";
        }
        CHECK_ULP_CLOSE(P0[1], P1[1], 3);
        CHECK_ULP_CLOSE(P0[2], P1[2], 3);
    }
}

// Linear precision: If all control points lie *equidistantly* on a line, then the Bezier curve falls on a line.
// See Bezier and B-spline techniques, Section 2.8, Remark 8.
template<typename Real>
void test_linear_precision()
{
    std::vector<std::array<Real, 3>> control_points(10);
    std::array<Real, 3> P0 = {1,1,1};
    std::array<Real, 3> Pf = {2,2,2};
    control_points[0] = P0;
    control_points[9] = Pf;
    for (size_t i = 1; i < 9; ++i) {
        Real t = Real(i)/(control_points.size()-1);
        control_points[i][0] = (1-t)*P0[0] + t*Pf[0];
        control_points[i][1] = (1-t)*P0[1] + t*Pf[1];
        control_points[i][2] = (1-t)*P0[2] + t*Pf[2];
    }

    auto bp = bezier_polynomial(std::move(control_points));
    for (Real t = 0; t < 1; t += Real(1)/32) {
        std::array<Real, 3> P;
        P[0] = (1-t)*P0[0] + t*Pf[0];
        P[1] = (1-t)*P0[1] + t*Pf[1];
        P[2] = (1-t)*P0[2] + t*Pf[2];

        auto computed = bp(t);
        CHECK_ULP_CLOSE(P[0], computed[0], 4);
        CHECK_ULP_CLOSE(P[1], computed[1], 4);
        CHECK_ULP_CLOSE(P[2], computed[2], 4);

        std::array<Real, 3> dP;
        dP[0] = Pf[0] - P0[0];
        dP[1] = Pf[1] - P0[1];
        dP[2] = Pf[2] - P0[2];
        auto dpComputed = bp.prime(t);
        CHECK_ULP_CLOSE(dP[0], dpComputed[0], 5);
    }
}

int main()
{
    #ifdef __STDCPP_FLOAT32_T__
    test_linear<std::float32_t>();
    test_quadratic<std::float32_t>();
    test_convex_hull<std::float32_t>();
    test_linear_precision<std::float32_t>();
    test_reversal_symmetry<std::float32_t>();
    #else
    test_linear<float>();
    test_quadratic<float>();
    test_convex_hull<float>();
    test_linear_precision<float>();
    test_reversal_symmetry<float>();
    #endif

    #ifdef __STDCPP_FLOAT64_T__
    test_linear<std::float64_t>();
    test_quadratic<std::float64_t>();
    test_convex_hull<std::float64_t>();
    test_linear_precision<std::float64_t>();
    test_reversal_symmetry<std::float64_t>();
    #else
    test_linear<double>();
    test_quadratic<double>();
    test_convex_hull<double>();
    test_linear_precision<double>();
    test_reversal_symmetry<double>();
    #endif

    #ifdef BOOST_HAS_FLOAT128
    test_linear<float128>();
    test_quadratic<float128>();
    test_convex_hull<float128>();
    #endif

    return boost::math::test::report_errors();
}

#else
int main() {
    return 0;
}
#endif
