//  (C) Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pch.hpp>

#define BOOST_TEST_MAIN

// See: https://github.com/boostorg/math/issues/1115
#if __has_include(<X11/X.h>)
#  include <X11/X.h>
#endif

#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/test/results_collector.hpp>
#include <boost/math/special_functions/beta.hpp>
#include <boost/math/distributions/skew_normal.hpp>
#include <boost/math/tools/polynomial.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/test/results_collector.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/array.hpp>
#include <boost/type_index.hpp>
#include "table_type.hpp"
#include <iostream>
#include <iomanip>

#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/cpp_complex.hpp>

#define BOOST_CHECK_CLOSE_EX(a, b, prec, i) \
   {\
      unsigned int failures = boost::unit_test::results_collector.results( boost::unit_test::framework::current_test_case().p_id ).p_assertions_failed;\
      BOOST_CHECK_CLOSE(a, b, prec); \
      if(failures != boost::unit_test::results_collector.results( boost::unit_test::framework::current_test_case().p_id ).p_assertions_failed)\
      {\
         std::cerr << "Failure was at row " << i << std::endl;\
         std::cerr << std::setprecision(35); \
         std::cerr << "{ " << data[i][0] << " , " << data[i][1] << " , " << data[i][2];\
         std::cerr << " , " << data[i][3] << " , " << data[i][4] << " , " << data[i][5] << " } " << std::endl;\
      }\
   }


//
// Implement various versions of inverse of the incomplete beta
// using different root finding algorithms, and deliberately "bad"
// starting conditions: that way we get all the pathological cases
// we could ever wish for!!!
//

template <class T, class Policy>
struct ibeta_roots_1   // for first order algorithms
{
   ibeta_roots_1(T _a, T _b, T t, bool inv = false, bool neg = false)
      : a(_a), b(_b), target(t), invert(inv), neg(neg) {}

   T operator()(const T& x)
   {
      return boost::math::detail::ibeta_imp(a, b, (neg ? -x : x), Policy(), invert, true) - target;
   }
private:
   T a, b, target;
   bool invert, neg;
};

template <class T, class Policy>
struct ibeta_roots_2   // for second order algorithms
{
   ibeta_roots_2(T _a, T _b, T t, bool inv = false, bool neg = false)
      : a(_a), b(_b), target(t), invert(inv), neg(neg) {}

   boost::math::tuple<T, T> operator()(const T& xx)
   {
      typedef typename boost::math::lanczos::lanczos<T, Policy>::type L;
      T x = neg ? -xx : xx;
      T f = boost::math::detail::ibeta_imp(a, b, x, Policy(), invert, true) - target;
      T f1 = invert ?
         -boost::math::detail::ibeta_power_terms(b, a, 1 - x, x, L(), true, Policy())
               : boost::math::detail::ibeta_power_terms(a, b, x, 1 - x, L(), true, Policy());
      T y = 1 - x;
      if(y == 0)
         y = boost::math::tools::min_value<T>() * 8;
      f1 /= y * x;

      // make sure we don't have a zero derivative:
      if(f1 == 0)
         f1 = (invert ? -1 : 1) * boost::math::tools::min_value<T>() * 64;

      return boost::math::make_tuple(f, neg ? -f1 : f1);
   }
private:
   T a, b, target;
   bool invert, neg;
};

template <class T, class Policy>
struct ibeta_roots_3   // for third order algorithms
{
   ibeta_roots_3(T _a, T _b, T t, bool inv = false, bool neg = false)
      : a(_a), b(_b), target(t), invert(inv), neg(neg) {}

   boost::math::tuple<T, T, T> operator()(const T& xx)
   {
      typedef typename boost::math::lanczos::lanczos<T, Policy>::type L;
      T x = neg ? -xx : xx;
      T f = boost::math::detail::ibeta_imp(a, b, x, Policy(), invert, true) - target;
      T f1 = invert ?
               -boost::math::detail::ibeta_power_terms(b, a, 1 - x, x, L(), true, Policy())
               : boost::math::detail::ibeta_power_terms(a, b, x, 1 - x, L(), true, Policy());
      T y = 1 - x;
      if(y == 0)
         y = boost::math::tools::min_value<T>() * 8;
      f1 /= y * x;
      T f2 = f1 * (-y * a + (b - 2) * x + 1) / (y * x);
      if(invert)
         f2 = -f2;

      // make sure we don't have a zero derivative:
      if(f1 == 0)
         f1 = (invert ? -1 : 1) * boost::math::tools::min_value<T>() * 64;

      if (neg)
      {
         f1 = -f1;
      }

      return boost::math::make_tuple(f, f1, f2);
   }
private:
   T a, b, target;
   bool invert, neg;
};

double inverse_ibeta_bisect(double a, double b, double z)
{
   typedef boost::math::policies::policy<> pol;
   bool invert = false;
   int bits = std::numeric_limits<double>::digits;

   //
   // special cases, we need to have these because there may be other
   // possible answers:
   //
   if(z == 1) return 1;
   if(z == 0) return 0;

   //
   // We need a good estimate of the error in the incomplete beta function
   // so that we don't set the desired precision too high.  Assume that 3-bits
   // are lost each time the arguments increase by a factor of 10:
   //
   using namespace std;
   int bits_lost = static_cast<int>(ceil(log10((std::max)(a, b)) * 3));
   if(bits_lost < 0)
      bits_lost = 3;
   else
      bits_lost += 3;
   int precision = bits - bits_lost;

   double min = 0;
   double max = 1;
   boost::math::tools::eps_tolerance<double> tol(precision);
   return boost::math::tools::bisect(ibeta_roots_1<double, pol>(a, b, z, invert), min, max, tol).first;
}

double inverse_ibeta_bisect_neg(double a, double b, double z)
{
   typedef boost::math::policies::policy<> pol;
   bool invert = false;
   int bits = std::numeric_limits<double>::digits;

   //
   // special cases, we need to have these because there may be other
   // possible answers:
   //
   if(z == 1) return 1;
   if(z == 0) return 0;

   //
   // We need a good estimate of the error in the incomplete beta function
   // so that we don't set the desired precision too high.  Assume that 3-bits
   // are lost each time the arguments increase by a factor of 10:
   //
   using namespace std;
   int bits_lost = static_cast<int>(ceil(log10((std::max)(a, b)) * 3));
   if(bits_lost < 0)
      bits_lost = 3;
   else
      bits_lost += 3;
   int precision = bits - bits_lost;

   double min = -1;
   double max = 0;
   boost::math::tools::eps_tolerance<double> tol(precision);
   return -boost::math::tools::bisect(ibeta_roots_1<double, pol>(a, b, z, invert, true), min, max, tol).first;
}

double inverse_ibeta_newton(double a, double b, double z)
{
   double guess = 0.5;
   bool invert = false;
   int bits = std::numeric_limits<double>::digits;

   //
   // special cases, we need to have these because there may be other
   // possible answers:
   //
   if(z == 1) return 1;
   if(z == 0) return 0;

   //
   // We need a good estimate of the error in the incomplete beta function
   // so that we don't set the desired precision too high.  Assume that 3-bits
   // are lost each time the arguments increase by a factor of 10:
   //
   using namespace std;
   int bits_lost = static_cast<int>(ceil(log10((std::max)(a, b)) * 3));
   if(bits_lost < 0)
      bits_lost = 3;
   else
      bits_lost += 3;
   int precision = bits - bits_lost;

   double min = 0;
   double max = 1;
   return boost::math::tools::newton_raphson_iterate(ibeta_roots_2<double, boost::math::policies::policy<> >(a, b, z, invert), guess, min, max, precision);
}

double inverse_ibeta_newton_neg(double a, double b, double z)
{
   double guess = 0.5;
   bool invert = false;
   int bits = std::numeric_limits<double>::digits;

   //
   // special cases, we need to have these because there may be other
   // possible answers:
   //
   if(z == 1) return 1;
   if(z == 0) return 0;

   //
   // We need a good estimate of the error in the incomplete beta function
   // so that we don't set the desired precision too high.  Assume that 3-bits
   // are lost each time the arguments increase by a factor of 10:
   //
   using namespace std;
   int bits_lost = static_cast<int>(ceil(log10((std::max)(a, b)) * 3));
   if(bits_lost < 0)
      bits_lost = 3;
   else
      bits_lost += 3;
   int precision = bits - bits_lost;

   double min = -1;
   double max = 0;
   return -boost::math::tools::newton_raphson_iterate(ibeta_roots_2<double, boost::math::policies::policy<> >(a, b, z, invert, true), -guess, min, max, precision);
}

double inverse_ibeta_halley(double a, double b, double z)
{
   double guess = 0.5;
   bool invert = false;
   int bits = std::numeric_limits<double>::digits;

   //
   // special cases, we need to have these because there may be other
   // possible answers:
   //
   if(z == 1) return 1;
   if(z == 0) return 0;

   //
   // We need a good estimate of the error in the incomplete beta function
   // so that we don't set the desired precision too high.  Assume that 3-bits
   // are lost each time the arguments increase by a factor of 10:
   //
   using namespace std;
   int bits_lost = static_cast<int>(ceil(log10((std::max)(a, b)) * 3));
   if(bits_lost < 0)
      bits_lost = 3;
   else
      bits_lost += 3;
   int precision = bits - bits_lost;

   double min = 0;
   double max = 1;
   return boost::math::tools::halley_iterate(ibeta_roots_3<double, boost::math::policies::policy<> >(a, b, z, invert), guess, min, max, precision);
}

double inverse_ibeta_halley_neg(double a, double b, double z)
{
   double guess = -0.5;
   bool invert = false;
   int bits = std::numeric_limits<double>::digits;

   //
   // special cases, we need to have these because there may be other
   // possible answers:
   //
   if(z == 1) return 1;
   if(z == 0) return 0;

   //
   // We need a good estimate of the error in the incomplete beta function
   // so that we don't set the desired precision too high.  Assume that 3-bits
   // are lost each time the arguments increase by a factor of 10:
   //
   using namespace std;
   int bits_lost = static_cast<int>(ceil(log10((std::max)(a, b)) * 3));
   if(bits_lost < 0)
      bits_lost = 3;
   else
      bits_lost += 3;
   int precision = bits - bits_lost;

   double min = -1;
   double max = 0;
   return -boost::math::tools::halley_iterate(ibeta_roots_3<double, boost::math::policies::policy<> >(a, b, z, invert, true), guess, min, max, precision);
}

double inverse_ibeta_schroder(double a, double b, double z)
{
   double guess = 0.5;
   bool invert = false;
   int bits = std::numeric_limits<double>::digits;

   //
   // special cases, we need to have these because there may be other
   // possible answers:
   //
   if(z == 1) return 1;
   if(z == 0) return 0;

   //
   // We need a good estimate of the error in the incomplete beta function
   // so that we don't set the desired precision too high.  Assume that 3-bits
   // are lost each time the arguments increase by a factor of 10:
   //
   using namespace std;
   int bits_lost = static_cast<int>(ceil(log10((std::max)(a, b)) * 3));
   if(bits_lost < 0)
      bits_lost = 3;
   else
      bits_lost += 3;
   int precision = bits - bits_lost;

   double min = 0;
   double max = 1;
   return boost::math::tools::schroder_iterate(ibeta_roots_3<double, boost::math::policies::policy<> >(a, b, z, invert), guess, min, max, precision);
}


template <class Real, class T>
void test_inverses(const T& data)
{
   using namespace std;
   typedef Real                   value_type;

   value_type precision = static_cast<value_type>(ldexp(1.0, 1-boost::math::policies::digits<value_type, boost::math::policies::policy<> >()/2)) * 150;
   if(boost::math::policies::digits<value_type, boost::math::policies::policy<> >() < 50)
      precision = 1;   // 1% or two decimal digits, all we can hope for when the input is truncated

   for(unsigned i = 0; i < data.size(); ++i)
   {
      //
      // These inverse tests are thrown off if the output of the
      // incomplete beta is too close to 1: basically there is insuffient
      // information left in the value we're using as input to the inverse
      // to be able to get back to the original value.
      //
      if(data[i][5] == 0)
      {
         BOOST_CHECK_EQUAL(inverse_ibeta_halley(Real(data[i][0]), Real(data[i][1]), Real(data[i][5])), value_type(0));
         BOOST_CHECK_EQUAL(inverse_ibeta_halley_neg(Real(data[i][0]), Real(data[i][1]), Real(data[i][5])), value_type(0));
         BOOST_CHECK_EQUAL(inverse_ibeta_schroder(Real(data[i][0]), Real(data[i][1]), Real(data[i][5])), value_type(0));
         BOOST_CHECK_EQUAL(inverse_ibeta_newton(Real(data[i][0]), Real(data[i][1]), Real(data[i][5])), value_type(0));
         BOOST_CHECK_EQUAL(inverse_ibeta_newton_neg(Real(data[i][0]), Real(data[i][1]), Real(data[i][5])), value_type(0));
         BOOST_CHECK_EQUAL(inverse_ibeta_bisect(Real(data[i][0]), Real(data[i][1]), Real(data[i][5])), value_type(0));
         BOOST_CHECK_EQUAL(inverse_ibeta_bisect_neg(Real(data[i][0]), Real(data[i][1]), Real(data[i][5])), value_type(0));
      }
      else if((1 - data[i][5] > 0.001)
         && (fabs(data[i][5]) > 2 * boost::math::tools::min_value<value_type>())
         && (fabs(data[i][5]) > 2 * boost::math::tools::min_value<double>()))
      {
         value_type inv = inverse_ibeta_halley(Real(data[i][0]), Real(data[i][1]), Real(data[i][5]));
         BOOST_CHECK_CLOSE_EX(Real(data[i][2]), inv, precision, i);
         inv = inverse_ibeta_halley_neg(Real(data[i][0]), Real(data[i][1]), Real(data[i][5]));
         BOOST_MATH_ASSERT(boost::math::isfinite(inv));
         BOOST_CHECK_CLOSE_EX(Real(data[i][2]), inv, precision, i);
         inv = inverse_ibeta_schroder(Real(data[i][0]), Real(data[i][1]), Real(data[i][5]));
         BOOST_CHECK_CLOSE_EX(Real(data[i][2]), inv, precision, i);
         inv = inverse_ibeta_newton(Real(data[i][0]), Real(data[i][1]), Real(data[i][5]));
         BOOST_CHECK_CLOSE_EX(Real(data[i][2]), inv, precision, i);
         inv = inverse_ibeta_newton_neg(Real(data[i][0]), Real(data[i][1]), Real(data[i][5]));
         BOOST_CHECK_CLOSE_EX(Real(data[i][2]), inv, precision, i);
         inv = inverse_ibeta_bisect(Real(data[i][0]), Real(data[i][1]), Real(data[i][5]));
         BOOST_CHECK_CLOSE_EX(Real(data[i][2]), inv, precision, i);
         inv = inverse_ibeta_bisect_neg(Real(data[i][0]), Real(data[i][1]), Real(data[i][5]));
         BOOST_CHECK_CLOSE_EX(Real(data[i][2]), inv, precision, i);
      }
      else if(1 == data[i][5])
      {
         BOOST_CHECK_EQUAL(inverse_ibeta_halley(Real(data[i][0]), Real(data[i][1]), Real(data[i][5])), value_type(1));
         BOOST_CHECK_EQUAL(inverse_ibeta_halley_neg(Real(data[i][0]), Real(data[i][1]), Real(data[i][5])), value_type(1));
         BOOST_CHECK_EQUAL(inverse_ibeta_schroder(Real(data[i][0]), Real(data[i][1]), Real(data[i][5])), value_type(1));
         BOOST_CHECK_EQUAL(inverse_ibeta_newton(Real(data[i][0]), Real(data[i][1]), Real(data[i][5])), value_type(1));
         BOOST_CHECK_EQUAL(inverse_ibeta_newton_neg(Real(data[i][0]), Real(data[i][1]), Real(data[i][5])), value_type(1));
         BOOST_CHECK_EQUAL(inverse_ibeta_bisect(Real(data[i][0]), Real(data[i][1]), Real(data[i][5])), value_type(1));
         BOOST_CHECK_EQUAL(inverse_ibeta_bisect_neg(Real(data[i][0]), Real(data[i][1]), Real(data[i][5])), value_type(1));
      }

   }
}

#ifndef SC_
#define SC_(x) static_cast<typename table_type<T>::type>(BOOST_JOIN(x, L))
#endif

template <class T>
void test_beta(T, const char* /* name */)
{
   //
   // The actual test data is rather verbose, so it's in a separate file
   //
   // The contents are as follows, each row of data contains
   // five items, input value a, input value b, integration limits x, beta(a, b, x) and ibeta(a, b, x):
   //
#  include "ibeta_small_data.ipp"

   test_inverses<T>(ibeta_small_data);

#  include "ibeta_data.ipp"

   test_inverses<T>(ibeta_data);

#  include "ibeta_large_data.ipp"

   test_inverses<T>(ibeta_large_data);
}

#if !defined(BOOST_NO_CXX11_AUTO_DECLARATIONS) && !defined(BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX) && !defined(BOOST_NO_CXX11_LAMBDAS)
template <class ComplexType>
void test_complex_newton()
{
    typedef typename ComplexType::value_type Real;
    std::cout << "Testing complex Newton's Method on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    using std::abs;
    using std::sqrt;
    using boost::math::tools::complex_newton;
    using boost::math::tools::polynomial;
    using boost::math::constants::half;

    Real tol = std::numeric_limits<Real>::epsilon();
    // p(z) = z^2 + 1, roots: \pm i.
    polynomial<ComplexType> p{{1,0}, {0, 0}, {1,0}};
    ComplexType guess{1,1};
    polynomial<ComplexType> p_prime = p.prime();
    auto f = [&](ComplexType z) { return std::make_pair<ComplexType, ComplexType>(p(z), p_prime(z)); };
    ComplexType root = complex_newton(f, guess);

    BOOST_CHECK(abs(root.real()) <= tol);
    BOOST_CHECK_CLOSE(root.imag(), (Real)1, tol);

    guess = -guess;
    root = complex_newton(f, guess);
    BOOST_CHECK(abs(root.real()) <= tol);
    BOOST_CHECK_CLOSE(root.imag(), (Real)-1, tol);

    // Test that double roots are handled correctly-as correctly as possible.
    // Convergence at a double root is not quadratic.
    // This sets p = (z-i)^2:
    p = polynomial<ComplexType>({{-1,0}, {0,-2}, {1,0}});
    p_prime = p.prime();
    guess = -guess;
    auto g = [&](ComplexType z) { return std::make_pair<ComplexType, ComplexType>(p(z), p_prime(z)); };
    root = complex_newton(g, guess);
    BOOST_CHECK(abs(root.real()) < 10*sqrt(tol));
    BOOST_CHECK_CLOSE(root.imag(), (Real)1, tol);

    // Test that zero derivatives are handled.
    // p(z) = z^2 + iz + 1
    p = polynomial<ComplexType>({{1,0}, {0,1}, {1,0}});
    // p'(z) = 2z + i
    p_prime = p.prime();
    guess = ComplexType(0,-boost::math::constants::half<Real>());
    auto g2 = [&](ComplexType z) { return std::make_pair<ComplexType, ComplexType>(p(z), p_prime(z)); };
    root = complex_newton(g2, guess);

    // Here's the other root, in case code changes cause it to be found:
    //ComplexType expected_root1{0, half<Real>()*(sqrt(static_cast<Real>(5)) - static_cast<Real>(1))};
    ComplexType expected_root2{0, -half<Real>()*(sqrt(static_cast<Real>(5)) + static_cast<Real>(1))};

    BOOST_CHECK_CLOSE(expected_root2.imag(),root.imag(), tol);
    BOOST_CHECK(abs(root.real()) < tol);

    // Does a zero root pass the termination criteria?
    p = polynomial<ComplexType>({{0,0}, {0,0}, {1,0}});
    p_prime = p.prime();
    guess = ComplexType(0, -boost::math::constants::half<Real>());
    auto g3 = [&](ComplexType z) { return std::make_pair<ComplexType, ComplexType>(p(z), p_prime(z)); };
    root = complex_newton(g3, guess);
    BOOST_CHECK(abs(root.real()) < tol);

    // Does a monstrous root pass?
    Real x = -pow(static_cast<Real>(10), 20);
    p = polynomial<ComplexType>({{x, x}, {1,0}});
    p_prime = p.prime();
    guess = ComplexType(0, -boost::math::constants::half<Real>());
    auto g4 = [&](ComplexType z) { return std::make_pair<ComplexType, ComplexType>(p(z), p_prime(z)); };
    root = complex_newton(g4, guess);
    BOOST_CHECK(abs(root.real() + x) < tol);
    BOOST_CHECK(abs(root.imag() + x) < tol);

}

// Polynomials which didn't factorize using Newton's method at first:
void test_daubechies_fails()
{
    std::cout << "Testing failures from Daubechies filter computation.\n";
    using std::abs;
    using std::sqrt;
    using boost::math::tools::complex_newton;
    using boost::math::tools::polynomial;
    using boost::math::constants::half;

    double tol = 500*std::numeric_limits<double>::epsilon();
    polynomial<std::complex<double>> p{{-185961388.136908293,141732493.98435241}, {601080390,0}};
    std::complex<double> guess{1,1};
    polynomial<std::complex<double>> p_prime = p.prime();
    auto f = [&](std::complex<double> z) { return std::make_pair<std::complex<double>, std::complex<double>>(p(z), p_prime(z)); };
    std::complex<double> root = complex_newton(f, guess);

    std::complex<double> expected_root = -p.data()[0]/p.data()[1];
    BOOST_CHECK_CLOSE(expected_root.imag(), root.imag(), tol);
    BOOST_CHECK_CLOSE(expected_root.real(), root.real(), tol);
}
#endif

#if !defined(BOOST_NO_CXX17_IF_CONSTEXPR)
template<class Real>
void test_solve_real_quadratic()
{
    #ifndef __CYGWIN__
    Real tol = std::numeric_limits<Real>::epsilon();
    #else
    Real tol = 2*std::numeric_limits<Real>::epsilon();
    #endif

    using boost::math::tools::quadratic_roots;
    auto [x0, x1] = quadratic_roots<Real>(1, 0, -1);
    BOOST_CHECK_CLOSE(x0, Real(-1), tol);
    BOOST_CHECK_CLOSE(x1, Real(1), tol);

    auto p = quadratic_roots((Real)7, (Real)0, (Real)0);
    BOOST_CHECK_SMALL(p.first, tol);
    BOOST_CHECK_SMALL(p.second, tol);

    // (x-7)^2 = x^2 - 14*x + 49:
    p = quadratic_roots((Real)1, (Real)-14, (Real)49);
    BOOST_CHECK_CLOSE(p.first, Real(7), tol);
    BOOST_CHECK_CLOSE(p.second, Real(7), tol);

    // The following tests do not pass with Cygwin as it makes no attempt at having a correct FMA
    // https://sourceware.org/git/gitweb.cgi?p=newlib-cygwin.git;a=blob;f=newlib/libm/common/s_fma.c
    #ifndef __CYGWIN__
    
    // This test does not pass in multiprecision,
    // due to the fact it does not have an fma:
    if (std::is_floating_point<Real>::value)
    {
        // (x-1)(x-1-eps) = x^2 + (-eps - 2)x + (1)(1+eps)
        Real eps = 2*std::numeric_limits<Real>::epsilon();
        Real b = 256 * (-2 - eps);
        Real c = 256 * (1 + eps);
        p = quadratic_roots((Real)256, b, c);
        BOOST_CHECK_CLOSE(p.first, Real(1), tol);
        BOOST_CHECK_CLOSE(p.second, Real(1) + eps, tol);
    }

    if (std::is_same<Real, double>::value)
    {
        // Kahan's example: This is the test that demonstrates the necessity of the fma instruction.
        // https://en.wikipedia.org/wiki/Loss_of_significance#Instability_of_the_quadratic_equation
        p = quadratic_roots<Real>((Real)94906265.625, (Real )-189812534, (Real)94906268.375);
        BOOST_CHECK_CLOSE_FRACTION(p.first, Real(1), tol);
        BOOST_CHECK_CLOSE_FRACTION(p.second, 1.000000028975958, 4*tol);
    }
    
    #endif // __CYGWIN__
}

template<class Z>
void test_solve_int_quadratic()
{
    double tol = std::numeric_limits<double>::epsilon();
    using boost::math::tools::quadratic_roots;
    auto [x0, x1] = quadratic_roots(1, 0, -1);
    BOOST_CHECK_CLOSE(x0, double(-1), tol);
    BOOST_CHECK_CLOSE(x1, double(1), tol);

    auto p = quadratic_roots(7, 0, 0);
    BOOST_CHECK_SMALL(p.first, tol);
    BOOST_CHECK_SMALL(p.second, tol);

    // (x-7)^2 = x^2 - 14*x + 49:
    p = quadratic_roots(1, -14, 49);
    BOOST_CHECK_CLOSE(p.first, double(7), tol);
    BOOST_CHECK_CLOSE(p.second, double(7), tol);
}

template<class ComplexType>
void test_solve_complex_quadratic()
{
    using Real = typename ComplexType::value_type;
    Real tol = std::numeric_limits<Real>::epsilon();
    using boost::math::tools::quadratic_roots;
    auto [x0, x1] = quadratic_roots<ComplexType>({1,0}, {0,0}, {-1,0});
    BOOST_CHECK_CLOSE(x0.real(), Real(-1), tol);
    BOOST_CHECK_CLOSE(x1.real(), Real(1), tol);
    BOOST_CHECK_SMALL(x0.imag(), tol);
    BOOST_CHECK_SMALL(x1.imag(), tol);

    auto p = quadratic_roots<ComplexType>({7,0}, {0,0}, {0,0});
    BOOST_CHECK_SMALL(p.first.real(), tol);
    BOOST_CHECK_SMALL(p.second.real(), tol);

    // (x-7)^2 = x^2 - 14*x + 49:
    p = quadratic_roots<ComplexType>({1,0}, {-14,0}, {49,0});
    BOOST_CHECK_CLOSE(p.first.real(), Real(7), tol);
    BOOST_CHECK_CLOSE(p.second.real(), Real(7), tol);

}

#endif

void test_failures()
{
#if !defined(BOOST_NO_CXX11_LAMBDAS)
   // There is no root:
   BOOST_CHECK_THROW(boost::math::tools::newton_raphson_iterate([](double x) { return std::make_pair(x * x + 1, 2 * x); }, 10.0, -12.0, 12.0, 52), boost::math::evaluation_error);
   BOOST_CHECK_THROW(boost::math::tools::newton_raphson_iterate([](double x) { return std::make_pair(x * x + 1, 2 * x); }, -10.0, -12.0, 12.0, 52), boost::math::evaluation_error);
   // There is a root, but a bad guess takes us into a local minima:
   BOOST_CHECK_THROW(boost::math::tools::newton_raphson_iterate([](double x) { return std::make_pair(boost::math::pow<6>(x) - 2 * boost::math::pow<4>(x) + x + 0.5, 6 * boost::math::pow<5>(x) - 8 * boost::math::pow<3>(x) + 1); }, 0.75, -20., 20., 52), boost::math::evaluation_error);

   // There is no root:
   BOOST_CHECK_THROW(boost::math::tools::halley_iterate([](double x) { return std::make_tuple(x * x + 1, 2 * x, 2); }, 10.0, -12.0, 12.0, 52), boost::math::evaluation_error);
   BOOST_CHECK_THROW(boost::math::tools::halley_iterate([](double x) { return std::make_tuple(x * x + 1, 2 * x, 2); }, -10.0, -12.0, 12.0, 52), boost::math::evaluation_error);
   // There is a root, but a bad guess takes us into a local minima:
   BOOST_CHECK_THROW(boost::math::tools::halley_iterate([](double x) { return std::make_tuple(boost::math::pow<6>(x) - 2 * boost::math::pow<4>(x) + x + 0.5, 6 * boost::math::pow<5>(x) - 8 * boost::math::pow<3>(x) + 1, 30 * boost::math::pow<4>(x) - 24 * boost::math::pow<2>(x)); }, 0.75, -20., 20., 52), boost::math::evaluation_error);
#endif
}

BOOST_AUTO_TEST_CASE( test_main )
{

   test_beta(0.1, "double");

   // bug reports:
   boost::math::skew_normal_distribution<> dist(2.0, 1.0, -2.5);
   BOOST_CHECK(boost::math::isfinite(quantile(dist, 0.075)));

#if !defined(BOOST_NO_CXX11_AUTO_DECLARATIONS) && !defined(BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX) && !defined(BOOST_NO_CXX11_LAMBDAS)
   test_complex_newton<std::complex<float>>();
   test_complex_newton<std::complex<double>>();
   test_complex_newton<boost::multiprecision::cpp_complex_100>();
   test_daubechies_fails();
#endif

#if !defined(BOOST_NO_CXX17_IF_CONSTEXPR)
    test_solve_real_quadratic<float>();
    test_solve_real_quadratic<double>();
    test_solve_real_quadratic<long double>();
    test_solve_real_quadratic<boost::multiprecision::cpp_bin_float_50>();

    test_solve_int_quadratic<int>();
    test_solve_complex_quadratic<std::complex<double>>();
#endif
    test_failures();
}
