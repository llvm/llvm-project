//  (C) Copyright Jeremy Murphy 2015.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/config.hpp>
#define BOOST_TEST_MAIN
#include <boost/array.hpp>
#include <boost/math/tools/polynomial.hpp>
#ifndef BOOST_MATH_STANDALONE
#include <boost/integer/common_factor_rt.hpp>
#endif
#include <boost/mpl/list.hpp>
#include <boost/mpl/joint_view.hpp>
#include <boost/test/unit_test.hpp>
#ifndef BOOST_MATH_STANDALONE
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#endif
#include <utility>
#include <array>
#include <list>

#if !defined(TEST1) && !defined(TEST2) && !defined(TEST3)
#  define TEST1
#  define TEST2
#  define TEST3
#endif

using namespace boost::math;
using boost::integer::gcd;
using namespace boost::math::tools;
using namespace std;
using boost::integer::gcd_detail::Euclid_gcd;
using boost::math::tools::subresultant_gcd;

template <typename T>
struct answer
{
    answer(std::pair< polynomial<T>, polynomial<T> > const &x) :
    quotient(x.first), remainder(x.second) {}

    polynomial<T> quotient;
    polynomial<T> remainder;
};

std::array<double, 4> const d3a = {{10, -6, -4, 3}};
std::array<double, 4> const d3b = {{-7, 5, 6, 1}};

std::array<double, 2> const d1a = {{-2, 1}};
std::array<double, 1> const d0a = {{6}};
std::array<double, 2> const d0a1 = {{0, 6}};
std::array<double, 6> const d0a5 = {{0, 0, 0, 0, 0, 6}};


std::array<int, 9> const d8 = {{-5, 2, 8, -3, -3, 0, 1, 0, 1}};
std::array<int, 9> const d8b = {{0, 2, 8, -3, -3, 0, 1, 0, 1}};



BOOST_AUTO_TEST_CASE(trivial)
{
   /* We have one empty test case here, so that there is always something for Boost.Test to do even if the tests below are #if'ed out */
}


#ifdef TEST1

std::array<double, 4> const d3c = {{10.0/3.0, -2.0, -4.0/3.0, 1.0}};
std::array<double, 3> const d2a = {{-2, 2, 3}};
std::array<double, 3> const d2b = {{-7, 5, 6}};
std::array<double, 3> const d2c = {{31, -21, -22}};
std::array<double, 1> const d0b = {{3}};
std::array<int, 7> const d6 = {{21, -9, -4, 0, 5, 0, 3}};
std::array<int, 3> const d2 = {{-6, 0, 9}};
std::array<int, 6> const d5 = {{-9, 0, 3, 0, -15}};


BOOST_AUTO_TEST_CASE( test_construction )
{
    polynomial<double> const a(d3a.begin(), d3a.end());
    polynomial<double> const b(d3a.begin(), 3);
    BOOST_CHECK_EQUAL(a, b);
}

#ifdef BOOST_MATH_HAS_IS_CONST_ITERABLE

#include <list>
#include <array>

BOOST_AUTO_TEST_CASE(test_range_construction)
{
   std::list<double> l{ 1, 2, 3, 4 };
   std::array<double, 4> a{ 3, 4, 5, 6 };
   polynomial<double> p1{ 1, 2, 3, 4 };
   polynomial<double> p2{ 3, 4, 5, 6 };

   polynomial<double> p3(l);
   polynomial<double> p4(a);

   BOOST_CHECK_EQUAL(p1, p3);
   BOOST_CHECK_EQUAL(p2, p4);
}
#endif

#if !defined(BOOST_NO_CXX11_HDR_INITIALIZER_LIST) && !BOOST_WORKAROUND(BOOST_GCC_VERSION, < 40500)
BOOST_AUTO_TEST_CASE( test_initializer_list_construction )
{
    polynomial<double> a(begin(d3a), end(d3a));
    polynomial<double> b = {10, -6, -4, 3};
    polynomial<double> c{10, -6, -4, 3};
    polynomial<double> d{10, -6, -4, 3, 0, 0};
    BOOST_CHECK_EQUAL(a, b);
    BOOST_CHECK_EQUAL(b, c);
    BOOST_CHECK_EQUAL(d.degree(), 3u);
}

BOOST_AUTO_TEST_CASE( test_initializer_list_assignment )
{
    polynomial<double> a(begin(d3a), end(d3a));
    polynomial<double> b;
    b = {10, -6, -4, 3, 0, 0};
    BOOST_CHECK_EQUAL(b.degree(), 3u);
    BOOST_CHECK_EQUAL(a, b);
}
#endif


BOOST_AUTO_TEST_CASE( test_degree )
{
    polynomial<double> const zero;
    polynomial<double> const a(d3a.begin(), d3a.end());
    BOOST_CHECK_THROW(zero.degree(), std::logic_error);
    BOOST_CHECK_EQUAL(a.degree(), 3u);
}


BOOST_AUTO_TEST_CASE( test_division_over_field )
{
    polynomial<double> const a(d3a.begin(), d3a.end());
    polynomial<double> const b(d1a.begin(), d1a.end());
    polynomial<double> const q(d2a.begin(), d2a.end());
    polynomial<double> const r(d0a.begin(), d0a.end());
    polynomial<double> const c(d3b.begin(), d3b.end());
    polynomial<double> const d(d2b.begin(), d2b.end());
    polynomial<double> const e(d2c.begin(), d2c.end());
    polynomial<double> const f(d0b.begin(), d0b.end());
    polynomial<double> const g(d3c.begin(), d3c.end());
    polynomial<double> const zero;
    polynomial<double> const one(1.0);

    answer<double> result = quotient_remainder(a, b);
    BOOST_CHECK_EQUAL(result.quotient, q);
    BOOST_CHECK_EQUAL(result.remainder, r);
    BOOST_CHECK_EQUAL(a, q * b + r); // Sanity check.

    result = quotient_remainder(a, c);
    BOOST_CHECK_EQUAL(result.quotient, f);
    BOOST_CHECK_EQUAL(result.remainder, e);
    BOOST_CHECK_EQUAL(a, f * c + e); // Sanity check.

    result = quotient_remainder(a, f);
    BOOST_CHECK_EQUAL(result.quotient, g);
    BOOST_CHECK_EQUAL(result.remainder, zero);
    BOOST_CHECK_EQUAL(a, g * f + zero); // Sanity check.
    // Check that division by a regular number gives the same result.
    BOOST_CHECK_EQUAL(a / 3.0, g);
    BOOST_CHECK_EQUAL(a % 3.0, zero);

    // Sanity checks.
    BOOST_CHECK_EQUAL(a / a, one);
    BOOST_CHECK_EQUAL(a % a, zero);
    // BOOST_CHECK_EQUAL(zero / zero, zero); // TODO
}

BOOST_AUTO_TEST_CASE( test_division_over_ufd )
{
    polynomial<int> const zero;
    polynomial<int> const one(1);
    polynomial<int> const aa(d8.begin(), d8.end());
    polynomial<int> const bb(d6.begin(), d6.end());
    polynomial<int> const q(d2.begin(), d2.end());
    polynomial<int> const r(d5.begin(), d5.end());

    answer<int> result = quotient_remainder(aa, bb);
    BOOST_CHECK_EQUAL(result.quotient, q);
    BOOST_CHECK_EQUAL(result.remainder, r);

    // Sanity checks.
    BOOST_CHECK_EQUAL(aa / aa, one);
    BOOST_CHECK_EQUAL(aa % aa, zero);
}

#endif

template <typename T>
struct FM2GP_Ex_8_3__1
{
    polynomial<T> x;
    polynomial<T> y;
    polynomial<T> z;

    FM2GP_Ex_8_3__1()
    {
        std::array<T, 5> const x_data = {{105, 278, -88, -56, 16}};
        std::array<T, 5> const y_data = {{70, 232, -44, -64, 16}};
        std::array<T, 3> const z_data = {{35, -24, 4}};
        x = polynomial<T>(x_data.begin(), x_data.end());
        y = polynomial<T>(y_data.begin(), y_data.end());
        z = polynomial<T>(z_data.begin(), z_data.end());
    }
};

template <typename T>
struct FM2GP_Ex_8_3__2
{
    polynomial<T> x;
    polynomial<T> y;
    polynomial<T> z;

    FM2GP_Ex_8_3__2()
    {
        std::array<T, 5> const x_data = {{1, -6, -8, 6, 7}};
        std::array<T, 5> const y_data = {{1, -5, -2, 15, 11}};
        std::array<T, 3> const z_data = {{1, 2, 1}};
        x = polynomial<T>(x_data.begin(), x_data.end());
        y = polynomial<T>(y_data.begin(), y_data.end());
        z = polynomial<T>(z_data.begin(), z_data.end());
    }
};


template <typename T>
struct FM2GP_mixed
{
    polynomial<T> x;
    polynomial<T> y;
    polynomial<T> z;

    FM2GP_mixed()
    {
        std::array<T, 4> const x_data = {{-2.2, -3.3, 0, 1}};
        std::array<T, 3> const y_data = {{-4.4, 0, 1}};
        std::array<T, 2> const z_data= {{-2, 1}};
        x = polynomial<T>(x_data.begin(), x_data.end());
        y = polynomial<T>(y_data.begin(), y_data.end());
        z = polynomial<T>(z_data.begin(), z_data.end());
    }
};


template <typename T>
struct FM2GP_trivial
{
    polynomial<T> x;
    polynomial<T> y;
    polynomial<T> z;

    FM2GP_trivial()
    {
        std::array<T, 4> const x_data = {{-2, -3, 0, 1}};
        std::array<T, 3> const y_data = {{-4, 0, 1}};
        std::array<T, 2> const z_data= {{-2, 1}};
        x = polynomial<T>(x_data.begin(), x_data.end());
        y = polynomial<T>(y_data.begin(), y_data.end());
        z = polynomial<T>(z_data.begin(), z_data.end());
    }
};

// Sanity checks to make sure I didn't break it.
#ifdef TEST1
typedef boost::mpl::list<signed char, short, int, long> integral_test_types;
typedef boost::mpl::list<int, long> large_integral_test_types;
typedef boost::mpl::list<> mp_integral_test_types;
#elif defined(TEST2)
typedef boost::mpl::list<
#if !BOOST_WORKAROUND(BOOST_MSVC, <= 1500) && !defined(BOOST_MATH_STANDALONE)
   boost::multiprecision::cpp_int
#endif
> integral_test_types;
typedef integral_test_types large_integral_test_types;
typedef large_integral_test_types mp_integral_test_types;
#elif defined(TEST3)
typedef boost::mpl::list<> large_integral_test_types;
typedef boost::mpl::list<> integral_test_types;
typedef large_integral_test_types mp_integral_test_types;
#endif

#ifdef TEST1
typedef boost::mpl::list<double, long double> non_integral_test_types;
#elif defined(TEST2)
typedef boost::mpl::list<
#if !BOOST_WORKAROUND(BOOST_MSVC, <= 1500) && !defined(BOOST_MATH_STANDALONE)
   boost::multiprecision::cpp_rational
#endif
> non_integral_test_types;
#elif defined(TEST3)
typedef boost::mpl::list<
#if !BOOST_WORKAROUND(BOOST_MSVC, <= 1500) && !defined(BOOST_MATH_STANDALONE)
   boost::multiprecision::cpp_bin_float_single, boost::multiprecision::cpp_dec_float_50
#endif
> non_integral_test_types;
#endif

typedef boost::mpl::joint_view<integral_test_types, non_integral_test_types> all_test_types;


template <typename T>
void normalize(polynomial<T> &p)
{
    if (leading_coefficient(p) < T(0))
        std::transform(p.data().begin(), p.data().end(), p.data().begin(), std::negate<T>());
}

/**
 * Note that we do not expect 'pure' gcd algorithms to normalize the result.
 * However, the usual public interface function gcd() will do that.
 */

BOOST_AUTO_TEST_SUITE(test_subresultant_gcd)

// This test is just to show that gcd<polynomial<T>>(u, v) is defined (and works) when T is integral and multiprecision.
BOOST_FIXTURE_TEST_CASE_TEMPLATE( gcd_interface, T, mp_integral_test_types, FM2GP_Ex_8_3__1<T> )
{
    typedef FM2GP_Ex_8_3__1<T> fixture_type;
    polynomial<T> w;
    w = gcd(fixture_type::x, fixture_type::y);
    normalize(w);
    BOOST_CHECK_EQUAL(w, fixture_type::z);
    w = gcd(fixture_type::y, fixture_type::x);
    normalize(w);
    BOOST_CHECK_EQUAL(w, fixture_type::z);
}

// This test is just to show that gcd<polynomial<T>>(u, v) is defined (and works) when T is floating point.
BOOST_FIXTURE_TEST_CASE_TEMPLATE( gcd_float_interface, T, non_integral_test_types, FM2GP_Ex_8_3__1<T> )
{
    typedef FM2GP_Ex_8_3__1<T> fixture_type;
    polynomial<T> w;
    w = gcd(fixture_type::x, fixture_type::y);
    normalize(w);
    BOOST_CHECK_EQUAL(w, fixture_type::z);
    w = gcd(fixture_type::y, fixture_type::x);
    normalize(w);
    BOOST_CHECK_EQUAL(w, fixture_type::z);
}

// The following tests call subresultant_gcd explicitly to remove any ambiguity
// and to permit testing on single-precision integral types.
BOOST_FIXTURE_TEST_CASE_TEMPLATE( Ex_8_3__1, T, large_integral_test_types, FM2GP_Ex_8_3__1<T> )
{
    typedef FM2GP_Ex_8_3__1<T> fixture_type;
    polynomial<T> w;
    w = subresultant_gcd(fixture_type::x, fixture_type::y);
    normalize(w);
    BOOST_CHECK_EQUAL(w, fixture_type::z);
    w = subresultant_gcd(fixture_type::y, fixture_type::x);
    normalize(w);
    BOOST_CHECK_EQUAL(w, fixture_type::z);
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE( Ex_8_3__2, T, large_integral_test_types, FM2GP_Ex_8_3__2<T> )
{
    typedef FM2GP_Ex_8_3__2<T> fixture_type;
    polynomial<T> w;
    w = subresultant_gcd(fixture_type::x, fixture_type::y);
    normalize(w);
    BOOST_CHECK_EQUAL(w, fixture_type::z);
    w = subresultant_gcd(fixture_type::y, fixture_type::x);
    normalize(w);
    BOOST_CHECK_EQUAL(w, fixture_type::z);
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE( trivial_int, T, large_integral_test_types, FM2GP_trivial<T> )
{
    typedef FM2GP_trivial<T> fixture_type;
    polynomial<T> w;
    w = subresultant_gcd(fixture_type::x, fixture_type::y);
    normalize(w);
    BOOST_CHECK_EQUAL(w, fixture_type::z);
    w = subresultant_gcd(fixture_type::y, fixture_type::x);
    normalize(w);
    BOOST_CHECK_EQUAL(w, fixture_type::z);
}

BOOST_AUTO_TEST_SUITE_END()


BOOST_AUTO_TEST_CASE_TEMPLATE( test_addition, T, all_test_types )
{
    polynomial<T> const a(d3a.begin(), d3a.end());
    polynomial<T> const b(d1a.begin(), d1a.end());
    polynomial<T> const zero;

    polynomial<T> result = a + b; // different degree
    std::array<T, 4> tmp = {{8, -5, -4, 3}};
    polynomial<T> expected(tmp.begin(), tmp.end());
    BOOST_CHECK_EQUAL(result, expected);
    BOOST_CHECK_EQUAL(a + zero, a);
    BOOST_CHECK_EQUAL(a + b, b + a);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( test_subtraction, T, all_test_types )
{
    polynomial<T> const a(d3a.begin(), d3a.end());
    polynomial<T> const zero;

    BOOST_CHECK_EQUAL(a - T(0), a);
    BOOST_CHECK_EQUAL(T(0) - a, -a);
    BOOST_CHECK_EQUAL(a - zero, a);
    BOOST_CHECK_EQUAL(zero - a, -a);
    BOOST_CHECK_EQUAL(a - a, zero);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( test_multiplication, T, all_test_types )
{
    polynomial<T> const a(d3a.begin(), d3a.end());
    polynomial<T> const b(d1a.begin(), d1a.end());
    polynomial<T> const zero;
    std::array<T, 7> const d3a_sq = {{100, -120, -44, 108, -20, -24, 9}};
    polynomial<T> const a_sq(d3a_sq.begin(), d3a_sq.end());

    BOOST_CHECK_EQUAL(a * T(0), zero);
    BOOST_CHECK_EQUAL(a * zero, zero);
    BOOST_CHECK_EQUAL(zero * T(0), zero);
    BOOST_CHECK_EQUAL(zero * zero, zero);
    BOOST_CHECK_EQUAL(a * b, b * a);
    polynomial<T> aa(a);
    aa *= aa;
    BOOST_CHECK_EQUAL(aa, a_sq);
    BOOST_CHECK_EQUAL(aa, a * a);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( test_arithmetic_relations, T, all_test_types )
{
    polynomial<T> const a(d8b.begin(), d8b.end());
    polynomial<T> const b(d1a.begin(), d1a.end());

    BOOST_CHECK_EQUAL(a * T(2), a + a);
    BOOST_CHECK_EQUAL(a - b, -b + a);
    BOOST_CHECK_EQUAL(a, (a * a) / a);
    BOOST_CHECK_EQUAL(a, (a / a) * a);
}


BOOST_AUTO_TEST_CASE_TEMPLATE(test_non_integral_arithmetic_relations, T, non_integral_test_types )
{
    polynomial<T> const a(d8b.begin(), d8b.end());
    polynomial<T> const b(d1a.begin(), d1a.end());

    BOOST_CHECK_EQUAL(a * T(0.5), a / T(2));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_cont_and_pp, T, integral_test_types)
{
    std::array<polynomial<T>, 4> const q={{
        polynomial<T>(d8.begin(), d8.end()),
        polynomial<T>(d8b.begin(), d8b.end()),
        polynomial<T>(d3a.begin(), d3a.end()),
        polynomial<T>(d3b.begin(), d3b.end())
    }};
    for (std::size_t i = 0; i < q.size(); i++)
    {
        BOOST_CHECK_EQUAL(q[i], content(q[i]) * primitive_part(q[i]));
        BOOST_CHECK_EQUAL(primitive_part(q[i]), primitive_part(q[i], content(q[i])));
    }

    polynomial<T> const zero;
    BOOST_CHECK_EQUAL(primitive_part(zero), zero);
    BOOST_CHECK_EQUAL(content(zero), T(0));
}

BOOST_AUTO_TEST_CASE_TEMPLATE( test_self_multiply_assign, T, all_test_types )
{
    polynomial<T> a(d3a.begin(), d3a.end());
    polynomial<T> const b(a);
    std::array<double, 7> const d3a_sq = {{100, -120, -44, 108, -20, -24, 9}};
    polynomial<T> const asq(d3a_sq.begin(), d3a_sq.end());

    a *= a;

    BOOST_CHECK_EQUAL(a, b*b);
    BOOST_CHECK_EQUAL(a, asq);

    a *= a;

    BOOST_CHECK_EQUAL(a, b*b*b*b);
}


BOOST_AUTO_TEST_CASE_TEMPLATE(test_right_shift, T, all_test_types )
{
    polynomial<T> a(d8b.begin(), d8b.end());
    polynomial<T> const aa(a);
    polynomial<T> const b(d8b.begin() + 1, d8b.end());
    polynomial<T> const c(d8b.begin() + 5, d8b.end());
    a >>= 0u;
    BOOST_CHECK_EQUAL(a, aa);
    a >>= 1u;
    BOOST_CHECK_EQUAL(a, b);
    a = a >> 4u;
    BOOST_CHECK_EQUAL(a, c);
}


BOOST_AUTO_TEST_CASE_TEMPLATE(test_left_shift, T, all_test_types )
{
    polynomial<T> a(d0a.begin(), d0a.end());
    polynomial<T> const aa(a);
    polynomial<T> const b(d0a1.begin(), d0a1.end());
    polynomial<T> const c(d0a5.begin(), d0a5.end());
    a <<= 0u;
    BOOST_CHECK_EQUAL(a, aa);
    a <<= 1u;
    BOOST_CHECK_EQUAL(a, b);
    a = a << 4u;
    BOOST_CHECK_EQUAL(a, c);
    polynomial<T> zero;
    // Multiplying zero by x should still be zero.
    zero <<= 1u;
    BOOST_CHECK_EQUAL(zero, zero_element(multiplies< polynomial<T> >()));
}


BOOST_AUTO_TEST_CASE_TEMPLATE(test_odd_even, T, all_test_types)
{
    polynomial<T> const zero;
    BOOST_CHECK_EQUAL(odd(zero), false);
    BOOST_CHECK_EQUAL(even(zero), true);
    polynomial<T> const a(d0a.begin(), d0a.end());
    BOOST_CHECK_EQUAL(odd(a), true);
    BOOST_CHECK_EQUAL(even(a), false);
    polynomial<T> const b(d0a1.begin(), d0a1.end());
    BOOST_CHECK_EQUAL(odd(b), false);
    BOOST_CHECK_EQUAL(even(b), true);
}

// NOTE: Slightly unexpected: this unit test passes even when T = char.
BOOST_AUTO_TEST_CASE_TEMPLATE( test_pow, T, all_test_types )
{
   if (std::numeric_limits<T>::digits < 32)
      return;   // Invokes undefined behaviour
    polynomial<T> a(d3a.begin(), d3a.end());
    polynomial<T> const one(T(1));
    std::array<double, 7> const d3a_sqr = {{100, -120, -44, 108, -20, -24, 9}};
    std::array<double, 10> const d3a_cub =
        {{1000, -1800, -120, 2124, -1032, -684, 638, -18, -108, 27}};
    polynomial<T> const asqr(d3a_sqr.begin(), d3a_sqr.end());
    polynomial<T> const acub(d3a_cub.begin(), d3a_cub.end());

    BOOST_CHECK_EQUAL(pow(a, 0), one);
    BOOST_CHECK_EQUAL(pow(a, 1), a);
    BOOST_CHECK_EQUAL(pow(a, 2), asqr);
    BOOST_CHECK_EQUAL(pow(a, 3), acub);
    BOOST_CHECK_EQUAL(pow(a, 4), pow(asqr, 2));
    BOOST_CHECK_EQUAL(pow(a, 5), asqr * acub);
    BOOST_CHECK_EQUAL(pow(a, 6), pow(acub, 2));
    BOOST_CHECK_EQUAL(pow(a, 7), acub * acub * a);

    BOOST_CHECK_THROW(pow(a, -1), std::domain_error);
    BOOST_CHECK_EQUAL(pow(one, 137), one);
}


BOOST_AUTO_TEST_CASE_TEMPLATE(test_bool, T, all_test_types)
{
    polynomial<T> const zero;
    polynomial<T> const a(d0a.begin(), d0a.end());
    BOOST_CHECK_EQUAL(bool(zero), false);
    BOOST_CHECK_EQUAL(bool(a), true);
}


BOOST_AUTO_TEST_CASE_TEMPLATE(test_set_zero, T, all_test_types)
{
    polynomial<T> const zero;
    polynomial<T> a(d0a.begin(), d0a.end());
    a.set_zero();
    BOOST_CHECK_EQUAL(a, zero);
    a.set_zero(); // Ensure that setting zero to zero is a no-op.
    BOOST_CHECK_EQUAL(a, zero);
}


BOOST_AUTO_TEST_CASE_TEMPLATE(test_leading_coefficient, T, all_test_types)
{
    polynomial<T> const zero;
    BOOST_CHECK_EQUAL(leading_coefficient(zero), T(0));
    polynomial<T> a(d0a.begin(), d0a.end());
    BOOST_CHECK_EQUAL(leading_coefficient(a), T(d0a.back()));
}

#if !defined(BOOST_NO_CXX11_RVALUE_REFERENCES) && !defined(BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX)
BOOST_AUTO_TEST_CASE_TEMPLATE(test_prime, T, all_test_types)
{
    std::vector<T> d{1,1,1,1,1};
    polynomial<T> p(std::move(d));
    polynomial<T> q = p.prime();
    BOOST_CHECK_EQUAL(q(0), T(1));

    for (size_t i = 0; i < q.size(); ++i)
    {
        BOOST_CHECK_EQUAL(q[i], i+1);
    }

    polynomial<T> P = p.integrate();
    BOOST_CHECK_EQUAL(P(0), T(0));
    for (size_t i = 1; i < P.size(); ++i)
    {
        BOOST_CHECK_EQUAL(P[i], 1/static_cast<T>(i));
    }

    polynomial<T> empty;
    q = empty.prime();
    BOOST_CHECK_EQUAL(q.size(), 0);

}
#endif
