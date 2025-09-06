/*
 * Copyright Nick Thompson, 2017
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <random>
#include <iostream>
#include <boost/type_index.hpp>
#include <boost/math/special_functions/chebyshev.hpp>
#include <boost/math/special_functions/sinc.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/array.hpp>
#include "math_unit_test.hpp"

using boost::multiprecision::cpp_bin_float_quad;
using boost::math::chebyshev_t;
using boost::math::chebyshev_t_prime;
using boost::math::chebyshev_u;

template<class Real>
void test_polynomials()
{
    std::cout << "Testing explicit polynomial representations of the Chebyshev polynomials on type " << boost::typeindex::type_id<Real>().pretty_name()  << "\n";

    Real x = -2;
    Real tol = 32*std::numeric_limits<Real>::epsilon();
    while (x < 2)
    {
        CHECK_MOLLIFIED_CLOSE(chebyshev_t(0, x), Real(1), tol);
        CHECK_MOLLIFIED_CLOSE(chebyshev_t(1, x), x, tol);
        CHECK_MOLLIFIED_CLOSE(chebyshev_t(2, x), 2*x*x - 1, tol);
        CHECK_MOLLIFIED_CLOSE(chebyshev_t(3, x), x*(4*x*x-3), tol);
        CHECK_MOLLIFIED_CLOSE(chebyshev_t(4, x), 8*x*x*(x*x - 1) + 1, tol);
        CHECK_MOLLIFIED_CLOSE(chebyshev_t(5, x), x*(16*x*x*x*x - 20*x*x + 5), tol);
        x += 1/static_cast<Real>(1<<7);
    }

    x = -2;
    while (x < 2)
    {
        CHECK_MOLLIFIED_CLOSE(chebyshev_u(0, x), Real(1), tol);
        CHECK_MOLLIFIED_CLOSE(chebyshev_u(1, x), 2*x, tol);
        CHECK_MOLLIFIED_CLOSE(chebyshev_u(2, x), 4*x*x - 1, tol);
        CHECK_MOLLIFIED_CLOSE(chebyshev_u(3, x), 4*x*(2*x*x - 1), tol);
        x += 1/static_cast<Real>(1<<7);
    }
}


template<class Real>
void test_derivatives()
{
    std::cout << "Testing explicit polynomial representations of the Chebyshev polynomial derivatives on type " << boost::typeindex::type_id<Real>().pretty_name()  << "\n";

    Real x = -2;
    Real tol = 4*std::numeric_limits<Real>::epsilon();
    while (x < 2)
    {
        CHECK_MOLLIFIED_CLOSE(chebyshev_t_prime(0, x), Real(0), tol);
        CHECK_MOLLIFIED_CLOSE(chebyshev_t_prime(1, x), Real(1), tol);
        CHECK_MOLLIFIED_CLOSE(chebyshev_t_prime(2, x), 4*x, tol);
        CHECK_MOLLIFIED_CLOSE(chebyshev_t_prime(3, x), 3*(4*x*x - 1), tol);
        CHECK_MOLLIFIED_CLOSE(chebyshev_t_prime(4, x), 16*x*(2*x*x - 1), tol);
        // This one makes the tolerance have to grow too large; the Chebyshev recurrence is more stable than naive polynomial evaluation anyway.
        //BOOST_CHECK_CLOSE_FRACTION(chebyshev_t_prime(5, x), 5*(4*x*x*(4*x*x - 3) + 1), tol);
        x += 1/static_cast<Real>(1<<7);
    }
}

template<class Real>
void test_clenshaw_recurrence()
{
    using boost::math::chebyshev_clenshaw_recurrence;
    std::array<Real, 5> c0 = { {2, 0, 0, 0, 0} };
    // Check the size = 1 case:
    std::array<Real, 1> c01 = { {2} };
    // Check the size = 2 case:
    std::array<Real, 2> c02 = { {2, 0} };
    std::array<Real, 4> c1 = { {0, 1, 0, 0} };
    std::array<Real, 4> c2 = { {0, 0, 1, 0} };
    std::array<Real, 5> c3 = { {0, 0, 0, 1, 0} };
    std::array<Real, 5> c4 = { {0, 0, 0, 0, 1} };
    std::array<Real, 6> c5 = { {0, 0, 0, 0, 0, 1} };
    std::array<Real, 7> c6 = { {0, 0, 0, 0, 0, 0, 1} };

    //
    // Error handling checks:
    //
    CHECK_THROW(chebyshev_clenshaw_recurrence(c0.data(), c0.size(), Real(-1), Real(1), Real(-2)), std::domain_error);
    CHECK_THROW(chebyshev_clenshaw_recurrence(c0.data(), c0.size(), Real(-1), Real(1), Real(2)), std::domain_error);
    CHECK_EQUAL(chebyshev_clenshaw_recurrence(c0.data(), 0, Real(-1), Real(1), Real(0.5)), Real(0));

    Real x = -1;
    // It's not clear from this test which one is more accurate; higher precision cast testing is required, and is done elsewhere:
    int ulps = 50;
    while (x <= 1)
    {
        Real y = chebyshev_clenshaw_recurrence(c0.data(), c0.size(), x);
        Real expected = chebyshev_t(0, x);
        CHECK_ULP_CLOSE(expected, y, ulps);
        y = chebyshev_clenshaw_recurrence(c0.data(), c0.size(), Real(-1), Real(1), x);
        CHECK_ULP_CLOSE(expected, y, ulps);

        y = chebyshev_clenshaw_recurrence(c01.data(), c01.size(), x);
        CHECK_ULP_CLOSE(expected, y, ulps);
        y = chebyshev_clenshaw_recurrence(c01.data(), c01.size(), Real(-1), Real(1), x);
        CHECK_ULP_CLOSE(expected, y, ulps);

        y = chebyshev_clenshaw_recurrence(c02.data(), c02.size(), x);
        CHECK_ULP_CLOSE(expected, y, ulps);
        y = chebyshev_clenshaw_recurrence(c02.data(), c02.size(), Real(-1), Real(1), x);
        CHECK_ULP_CLOSE(expected, y, ulps);

        expected = chebyshev_t(1,x);
        y = chebyshev_clenshaw_recurrence(c1.data(), c1.size(), x);
        CHECK_ULP_CLOSE(expected, y, ulps);
        y = chebyshev_clenshaw_recurrence(c1.data(), c1.size(), Real(-1), Real(1), x);
        CHECK_ULP_CLOSE(expected, y, ulps);

        expected = chebyshev_t(2, x);
        y = chebyshev_clenshaw_recurrence(c2.data(), c2.size(), x);
        CHECK_ULP_CLOSE(expected, y, ulps);
        y = chebyshev_clenshaw_recurrence(c2.data(), c2.size(), Real(-1), Real(1), x);
        CHECK_ULP_CLOSE(expected, y, ulps);

        expected = chebyshev_t(3, x);
        y = chebyshev_clenshaw_recurrence(c3.data(), c3.size(), x);
        CHECK_ULP_CLOSE(expected, y, ulps);
        y = chebyshev_clenshaw_recurrence(c3.data(), c3.size(), Real(-1), Real(1), x);
        CHECK_ULP_CLOSE(expected, y, ulps);

        expected = chebyshev_t(4, x);
        y = chebyshev_clenshaw_recurrence(c4.data(), c4.size(), x);
        CHECK_ULP_CLOSE(expected, y, ulps);
        y = chebyshev_clenshaw_recurrence(c4.data(), c4.size(), Real(-1), Real(1), x);
        CHECK_ULP_CLOSE(expected, y, ulps);

        expected = chebyshev_t(5, x);
        y = chebyshev_clenshaw_recurrence(c5.data(), c5.size(), x);
        CHECK_ULP_CLOSE(expected, y, ulps);
        y = chebyshev_clenshaw_recurrence(c5.data(), c5.size(), Real(-1), Real(1), x);
        CHECK_ULP_CLOSE(expected, y, ulps);

        expected = chebyshev_t(6, x);
        y = chebyshev_clenshaw_recurrence(c6.data(), c6.size(), x);
        CHECK_ULP_CLOSE(expected, y, ulps);
        y = chebyshev_clenshaw_recurrence(c6.data(), c6.size(), Real(-1), Real(1), x);
        CHECK_ULP_CLOSE(expected, y, ulps);

        x += static_cast<Real>(1)/static_cast<Real>(1 << 7);
    }
}

template<typename Real>
void test_translated_clenshaw_recurrence()
{
    using boost::math::chebyshev_clenshaw_recurrence;
    std::mt19937_64 mt(123242);
    std::uniform_real_distribution<Real> dis(-1,1);

    std::vector<Real> c(32);
    for (auto & d : c) {
        d = dis(mt);
    }
    int samples = 0;
    while (samples++ < 250) {
        Real x = dis(mt);
        Real expected = chebyshev_clenshaw_recurrence(c.data(), c.size(), x);
        // The computed value, in this case, should actually be better, since it has more information to use.
        // In any case, this test doesn't show Reinch's modification to the Clenshaw recurrence is better;
        // it shows they are doing roughly the same thing.
        Real computed = chebyshev_clenshaw_recurrence(c.data(), c.size(), Real(-1), Real(1), x);
        if (!CHECK_ULP_CLOSE(expected, computed, 1000)) {
            std::cerr << "  Problem occurred at x = " << x << "\n";
        }
    }
}

int main()
{
    test_polynomials<float>();
    test_polynomials<double>();
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    test_polynomials<long double>();
#endif
    test_polynomials<cpp_bin_float_quad>();

    test_derivatives<float>();
    test_derivatives<double>();
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    test_derivatives<long double>();
#endif
    test_derivatives<cpp_bin_float_quad>();

    test_clenshaw_recurrence<float>();
    test_clenshaw_recurrence<double>();
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    test_clenshaw_recurrence<long double>();
#endif

    test_translated_clenshaw_recurrence<double>();
    return boost::math::test::report_errors();
}
