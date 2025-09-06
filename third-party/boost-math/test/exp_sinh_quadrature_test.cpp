// Copyright Nick Thompson, 2017
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_TEST_MODULE exp_sinh_quadrature_test

#include <complex>
#include <type_traits>
#include <boost/math/tools/config.hpp>
#include <boost/math/tools/test_value.hpp>
#include <boost/multiprecision/cpp_complex.hpp>
#include <boost/math/concepts/real_concept.hpp>
#include <boost/test/included/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/quadrature/exp_sinh.hpp>
#include <boost/math/special_functions/sinc.hpp>
#include <boost/math/special_functions/bessel.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/math/special_functions/next.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/sinc.hpp>
#include <boost/type_traits/is_class.hpp>
#include <boost/type_index.hpp>

#ifdef BOOST_HAS_FLOAT128
#include <boost/multiprecision/complex128.hpp>
#endif

#if __has_include(<stdfloat>)
#  include <stdfloat>
#endif

using std::exp;
using std::cos;
using std::tan;
using std::log;
using std::sqrt;
using std::abs;
using std::sinh;
using std::cosh;
using std::pow;
using std::atan;
using boost::multiprecision::cpp_bin_float_50;
using boost::multiprecision::cpp_bin_float_100;
using boost::multiprecision::cpp_bin_float_quad;
using boost::math::constants::pi;
using boost::math::constants::half_pi;
using boost::math::constants::two_div_pi;
using boost::math::constants::half;
using boost::math::constants::third;
using boost::math::constants::half;
using boost::math::constants::third;
using boost::math::constants::catalan;
using boost::math::constants::ln_two;
using boost::math::constants::root_two;
using boost::math::constants::root_two_pi;
using boost::math::constants::root_pi;
using boost::math::quadrature::exp_sinh;

#if !defined(TEST1) && !defined(TEST2) && !defined(TEST3) && !defined(TEST4) && !defined(TEST5) && !defined(TEST6) && !defined(TEST7) && !defined(TEST8) && !defined(TEST9) && !defined(TEST10)
#  define TEST1
#  define TEST2
#  define TEST3
#  define TEST4
#  define TEST5
#  define TEST6
#  define TEST7
#  define TEST8
#  define TEST9
#  define TEST10
#endif

#ifdef _MSC_VER
#pragma warning (disable:4127)
#endif

//
// Coefficient generation code:
//
template <class T>
void print_levels(const T& v, const char* suffix)
{
   std::cout << "{\n";
   for (unsigned i = 0; i < v.size(); ++i)
   {
      std::cout << "      { ";
      for (unsigned j = 0; j < v[i].size(); ++j)
      {
         std::cout << v[i][j] << suffix << ", ";
      }
      std::cout << "},\n";
   }
   std::cout << "   };\n";
}

template <class T>
void print_levels(const std::pair<T, T>& p, const char* suffix = "")
{
   std::cout << "   static const std::vector<std::vector<Real> > abscissa = ";
   print_levels(p.first, suffix);
   std::cout << "   static const std::vector<std::vector<Real> > weights = ";
   print_levels(p.second, suffix);
}

template <class Real, class TargetType>
std::pair<std::vector<std::vector<Real>>, std::vector<std::vector<Real>> > generate_constants(unsigned max_rows)
{
   using boost::math::constants::half_pi;
   using boost::math::constants::two_div_pi;
   using boost::math::constants::pi;
   auto g = [](Real t)->Real { return exp(half_pi<Real>()*sinh(t)); };
   auto w = [](Real t)->Real { return cosh(t)*half_pi<Real>()*exp(half_pi<Real>()*sinh(t)); };

   std::vector<std::vector<Real>> abscissa, weights;

   std::vector<Real> temp;

   Real tmp = (Real(boost::math::tools::log_min_value<TargetType>()) + log(Real(boost::math::tools::epsilon<TargetType>())))*0.5f;
   Real t_min = asinh(two_div_pi<Real>()*tmp);
   // truncate t_min to an exact binary value:
   t_min = floor(t_min * 128) / 128;

   std::cout << "m_t_min = " << t_min << ";\n";

   // t_max is chosen to make g'(t_max) ~ sqrt(max) (g' grows faster than g).
   // This will allow some flexibility on the users part; they can at least square a number function without overflow.
   // But there is no unique choice; the further out we can evaluate the function, the better we can do on slowly decaying integrands.
   const Real t_max = log(2 * two_div_pi<Real>()*log(2 * two_div_pi<Real>()*sqrt(Real(boost::math::tools::max_value<TargetType>()))));

   Real h = 1;
   for (Real t = t_min; t < t_max; t += h)
   {
      temp.push_back(g(t));
   }
   abscissa.push_back(temp);
   temp.clear();

   for (Real t = t_min; t < t_max; t += h)
   {
      temp.push_back(w(t * h));
   }
   weights.push_back(temp);
   temp.clear();

   for (unsigned row = 1; row < max_rows; ++row)
   {
      h /= 2;
      for (Real t = t_min + h; t < t_max; t += 2 * h)
         temp.push_back(g(t));
      abscissa.push_back(temp);
      temp.clear();
   }
   h = 1;
   for (unsigned row = 1; row < max_rows; ++row)
   {
      h /= 2;
      for (Real t = t_min + h; t < t_max; t += 2 * h)
         temp.push_back(w(t));
      weights.push_back(temp);
      temp.clear();
   }

   return std::make_pair(abscissa, weights);
}


template <class Real>
const exp_sinh<Real>& get_integrator()
{
   static const exp_sinh<Real> integrator(14);
   return integrator;
}

template <class Real>
Real get_convergence_tolerance()
{
   return boost::math::tools::root_epsilon<Real>();
}

template<class Real>
void test_right_limit_infinite()
{
    std::cout << "Testing right limit infinite for tanh_sinh in 'A Comparison of Three High Precision Quadrature Schemes' on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    Real tol = 10 * boost::math::tools::epsilon<Real>();
    Real Q;
    Real Q_expected;
    Real error;
    Real L1;
    const auto& integrator = get_integrator<Real>();

    // Example 12
    const auto f2 = [](const Real& t)->Real { return exp(-t)/sqrt(t); };
    Q = integrator.integrate(f2, get_convergence_tolerance<Real>(), &error, &L1);
    Q_expected = root_pi<Real>();
    Real tol_mult = 1;
    // Multiprecision type have higher error rates, probably evaluation of f() is less accurate:
    if (!std::numeric_limits<Real>::digits10 || (std::numeric_limits<Real>::digits10 > 25))
       tol_mult = 1200;
    else if (std::numeric_limits<Real>::digits10 > std::numeric_limits<double>::digits10)
       tol_mult = 5;
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol * tol_mult);
    // The integrand is strictly positive, so it coincides with the value of the integral:
    BOOST_CHECK_CLOSE_FRACTION(L1, Q_expected, tol * tol_mult);

    #ifdef BOOST_MATH_STANDALONE
    BOOST_IF_CONSTEXPR (std::is_fundamental<Real>::value)
    #endif
    {
        auto f3 = [](Real t)->Real { Real z = exp(-t); if (z == 0) { return z; } return z*cos(t); };
        Q = integrator.integrate(f3, get_convergence_tolerance<Real>(), &error, &L1);
        Q_expected = half<Real>();
        BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
        Q = integrator.integrate(f3, 10, std::numeric_limits<Real>::has_infinity ? std::numeric_limits<Real>::infinity() : boost::math::tools::max_value<Real>(), get_convergence_tolerance<Real>(), &error, &L1);
        Q_expected = BOOST_MATH_TEST_VALUE(Real, -6.6976341310426674140007086979326069121526743314567805278252392932e-6);
        BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, 10 * tol);
        // Integrating through zero risks precision loss:
        Q = integrator.integrate(f3, -10, std::numeric_limits<Real>::has_infinity ? std::numeric_limits<Real>::infinity() : boost::math::tools::max_value<Real>(), get_convergence_tolerance<Real>(), &error, &L1);
        Q_expected = BOOST_MATH_TEST_VALUE(Real, -15232.3213626280525704332288302799653087046646639974940243044623285817777006);
        BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, std::numeric_limits<Real>::digits10 > 30 ? 1000 * tol : tol);

        auto f4 = [](Real t)->Real { return 1/(1+t*t); };
        Q = integrator.integrate(f4, 1, std::numeric_limits<Real>::has_infinity ? std::numeric_limits<Real>::infinity() : boost::math::tools::max_value<Real>(), get_convergence_tolerance<Real>(), &error, &L1);
        Q_expected = pi<Real>()/4;
        BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
        BOOST_CHECK_CLOSE_FRACTION(L1, Q_expected, tol);
        Q = integrator.integrate(f4, 20, std::numeric_limits<Real>::has_infinity ? std::numeric_limits<Real>::infinity() : boost::math::tools::max_value<Real>(), get_convergence_tolerance<Real>(), &error, &L1);
        Q_expected = BOOST_MATH_TEST_VALUE(Real, 0.0499583957219427614100062870348448814912770804235071744108534548299835954767);
        BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
        BOOST_CHECK_CLOSE_FRACTION(L1, Q_expected, tol);
        Q = integrator.integrate(f4, 500, std::numeric_limits<Real>::has_infinity ? std::numeric_limits<Real>::infinity() : boost::math::tools::max_value<Real>(), get_convergence_tolerance<Real>(), &error, &L1);
        Q_expected = BOOST_MATH_TEST_VALUE(Real, 0.0019999973333397333150476759363217553199063513829126652556286269630);
        BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
        BOOST_CHECK_CLOSE_FRACTION(L1, Q_expected, tol);
    }
}

template<class Real>
void test_left_limit_infinite()
{
    std::cout << "Testing left limit infinite for 1/(1+t^2) on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    Real tol = 10 * boost::math::tools::epsilon<Real>();
    Real Q;
    Real Q_expected;
    Real error;
    Real L1;
    const auto& integrator = get_integrator<Real>();

    // Example 11:
    #ifdef BOOST_MATH_STANDALONE
    BOOST_IF_CONSTEXPR (std::is_fundamental<Real>::value)
    #endif
    {
        auto f1 = [](const Real& t)->Real { return 1/(1+t*t);};
        Q = integrator.integrate(f1, std::numeric_limits<Real>::has_infinity ? -std::numeric_limits<Real>::infinity() : -boost::math::tools::max_value<Real>(), 0, get_convergence_tolerance<Real>(), &error, &L1);
        Q_expected = half_pi<Real>();
        BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
        BOOST_CHECK_CLOSE_FRACTION(L1, Q_expected, tol);
        Q = integrator.integrate(f1, std::numeric_limits<Real>::has_infinity ? -std::numeric_limits<Real>::infinity() : -boost::math::tools::max_value<Real>(), -20, get_convergence_tolerance<Real>(), &error, &L1);
        Q_expected = BOOST_MATH_TEST_VALUE(Real, 0.0499583957219427614100062870348448814912770804235071744108534548299835954767);
        BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
        BOOST_CHECK_CLOSE_FRACTION(L1, Q_expected, tol);
        Q = integrator.integrate(f1, std::numeric_limits<Real>::has_infinity ? -std::numeric_limits<Real>::infinity() : -boost::math::tools::max_value<Real>(), -500, get_convergence_tolerance<Real>(), &error, &L1);
        Q_expected = BOOST_MATH_TEST_VALUE(Real, 0.0019999973333397333150476759363217553199063513829126652556286269630);
        BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
        BOOST_CHECK_CLOSE_FRACTION(L1, Q_expected, tol);
    }
}


// Some examples of tough integrals from NR, section 4.5.4:
template<class Real>
void test_nr_examples()
{
    using std::sin;
    using std::cos;
    using std::pow;
    using std::exp;
    using std::sqrt;
    std::cout << "Testing type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    Real tol = 10 * boost::math::tools::epsilon<Real>();
    std::cout << std::setprecision(std::numeric_limits<Real>::digits10);
    Real Q;
    Real Q_expected;
    Real L1;
    Real error;
    const auto& integrator = get_integrator<Real>();

    auto f0 = [] (Real)->Real { return (Real) 0; };
    Q = integrator.integrate(f0, get_convergence_tolerance<Real>(), &error, &L1);
    Q_expected = 0;
    BOOST_CHECK_CLOSE_FRACTION(Q, 0.0f, tol);
    BOOST_CHECK_CLOSE_FRACTION(L1, 0.0f, tol);

    auto f = [](const Real& x)->Real { return 1/(1+x*x); };
    Q = integrator.integrate(f, get_convergence_tolerance<Real>(), &error, &L1);
    Q_expected = half_pi<Real>();
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
    BOOST_CHECK_CLOSE_FRACTION(L1, Q_expected, tol);

    auto f1 = [](Real x)->Real {
        Real z1 = exp(-x);
        if (z1 == 0)
        {
            return (Real) 0;
        }
        Real z2 = pow(x, -3*half<Real>())*z1;
        if (z2 == 0)
        {
            return (Real) 0;
        }
        return sin(x*half<Real>())*z2;
    };

    Q = integrator.integrate(f1, get_convergence_tolerance<Real>(), &error, &L1);
    Q_expected = sqrt(pi<Real>()*(sqrt((Real) 5) - 2));

    // The integrand is oscillatory; the accuracy is low.
    Real tol_mul = 1;
    if (std::numeric_limits<Real>::digits10 > 40)
       tol_mul = 500000;
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol_mul * tol);

    auto f2 = [](Real x)->Real { return x > boost::math::tools::log_max_value<Real>() ? Real(0) : Real(pow(x, -(Real) 2/(Real) 7)*exp(-x*x)); };
    Q = integrator.integrate(f2, get_convergence_tolerance<Real>(), &error, &L1);
    Q_expected = half<Real>()*boost::math::tgamma((Real) 5/ (Real) 14);
    tol_mul = 1;
    if ((std::numeric_limits<Real>::is_specialized == false) || (std::numeric_limits<Real>::digits10 > 40))
       tol_mul = 500;
    else
       tol_mul = 3;
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol_mul * tol);
    BOOST_CHECK_CLOSE_FRACTION(L1, Q_expected, tol_mul * tol);

    auto f3 = [](Real x)->Real { return (Real) 1/ (sqrt(x)*(1+x)); };
    Q = integrator.integrate(f3, get_convergence_tolerance<Real>(), &error, &L1);
    Q_expected = pi<Real>();

    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, 10*boost::math::tools::epsilon<Real>());
    BOOST_CHECK_CLOSE_FRACTION(L1, Q_expected, 10*boost::math::tools::epsilon<Real>());

    auto f4 = [](const Real& t)->Real { return  t > boost::math::tools::log_max_value<Real>() ? Real(0) : Real(exp(-t*t*half<Real>())); };
    Q = integrator.integrate(f4, get_convergence_tolerance<Real>(), &error, &L1);
    Q_expected = root_two_pi<Real>()/2;
    tol_mul = 1;
    if (std::numeric_limits<Real>::digits10 > 40)
       tol_mul = 5000;
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol_mul * tol);
    BOOST_CHECK_CLOSE_FRACTION(L1, Q_expected, tol_mul * tol);

    auto f5 = [](const Real& t)->Real { return 1/cosh(t);};
    Q = integrator.integrate(f5, get_convergence_tolerance<Real>(), &error, &L1);
    Q_expected = half_pi<Real>();
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol * 12);   // Fails at float precision without higher error rate
    BOOST_CHECK_CLOSE_FRACTION(L1, Q_expected, tol * 12);
}

// Definite integrals found in the CRC Handbook of Mathematical Formulas
template<class Real>
void test_crc()
{
    using std::sin;
    using std::pow;
    using std::exp;
    using std::sqrt;
    using std::log;
    using std::cos;
    std::cout << "Testing integral from CRC handbook on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    Real tol = 10 * boost::math::tools::epsilon<Real>();
    std::cout << std::setprecision(std::numeric_limits<Real>::digits10);
    Real Q;
    Real Q_expected;
    Real L1;
    Real error;
    const auto& integrator = get_integrator<Real>();

    auto f0 = [](const Real& x)->Real { return x > boost::math::tools::log_max_value<Real>() ? Real(0) : Real(log(x)*exp(-x)); };
    Q = integrator.integrate(f0, get_convergence_tolerance<Real>(), &error, &L1);
    Q_expected = -boost::math::constants::euler<Real>();
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);

    // Test the integral representation of the gamma function:
    auto f1 = [](Real t)->Real { Real x = exp(-t);
        if(x == 0)
        {
            return (Real) 0;
        }
        return pow(t, (Real) 12 - 1)*x;
    };

    Q = integrator.integrate(f1, get_convergence_tolerance<Real>(), &error, &L1);
    Q_expected = boost::math::tgamma(12.0f);
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);

    // Integral representation of the modified bessel function:
    // K_5(12)
    auto f2 = [](Real t)->Real {
        Real x = 12*cosh(t);
        if (x > boost::math::tools::log_max_value<Real>())
        {
            return (Real) 0;
        }
        return exp(-x)*cosh(5*t);
    };
    Q = integrator.integrate(f2, get_convergence_tolerance<Real>(), &error, &L1);
    Q_expected = boost::math::cyl_bessel_k<int, Real>(5, 12);
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
    // Laplace transform of cos(at)
    Real a = 20;
    Real s = 1;
    auto f3 = [&](Real t)->Real {
        Real x = s * t;
        if (x > boost::math::tools::log_max_value<Real>())
        {
            return (Real) 0;
        }
        return cos(a * t) * exp(-x);
    };

    // Since the integrand is oscillatory, we increase the tolerance:
    Real tol_mult = 10;
    // Multiprecision type have higher error rates, probably evaluation of f() is less accurate:
    if (!std::is_class<Real>::value)
    {
       // For high oscillation frequency, the quadrature sum is ill-conditioned.
       Q = integrator.integrate(f3, get_convergence_tolerance<Real>(), &error, &L1);
       Q_expected = s/(a*a+s*s);
       if (std::numeric_limits<Real>::digits10 > std::numeric_limits<double>::digits10)
          tol_mult = 500000; // we should really investigate this more??
       BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol_mult*tol);
    }

    //
    // This one doesn't pass for real_concept..
    //
    if (std::numeric_limits<Real>::is_specialized)
    {
       // Laplace transform of J_0(t):
       auto f4 = [&](Real t)->Real {
          Real x = s * t;
          if (x > boost::math::tools::log_max_value<Real>())
          {
             return (Real)0;
          }
          return boost::math::cyl_bessel_j(0, t) * exp(-x);
       };

       Q = integrator.integrate(f4, get_convergence_tolerance<Real>(), &error, &L1);
       Q_expected = 1 / sqrt(1 + s*s);
       tol_mult = 3;
       // Multiprecision type have higher error rates, probably evaluation of f() is less accurate:
       if ((std::numeric_limits<Real>::digits10 > std::numeric_limits<long double>::digits10) || (std::numeric_limits<Real>::digits > 100) || !std::numeric_limits<Real>::digits)
          tol_mult = 50000;
       BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol_mult * tol);
    }
    auto f6 = [](const Real& t)->Real { return t > boost::math::tools::log_max_value<Real>() ? Real(0) : Real(exp(-t*t)*log(t));};
    Q = integrator.integrate(f6, get_convergence_tolerance<Real>(), &error, &L1);
    Q_expected = -boost::math::constants::root_pi<Real>()*(boost::math::constants::euler<Real>() + 2*ln_two<Real>())/4;
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);

    // CRC Section 5.5, integral 591
    // The parameter p allows us to control the strength of the singularity.
    // Rapid convergence is not guaranteed for this function, as the branch cut makes it non-analytic on a disk.
    // This converges only when our test type has an extended exponent range as all the area of the integral
    // occurs so close to 0 (or 1) that we need abscissa values exceptionally small to find it.
    // "There's a lot of room at the bottom".
    // This version is transformed via argument substitution (exp(-x) for x) so that the integral is spread
    // over (0, INF).
    tol *= boost::math::tools::digits<Real>() > 100 ? 100000 : 75;
    for (Real pn = 99; pn > 0; pn -= 10) {
       Real p = pn / 100;
       auto f = [&](Real x)->Real
       {
          return x > 1000 * boost::math::tools::log_max_value<Real>() ? Real(0) : Real(exp(-x * (1 - p) + p * log(-boost::math::expm1(-x))));
       };
       Q = integrator.integrate(f, get_convergence_tolerance<Real>(), &error, &L1);
       Q_expected = 1 / boost::math::sinc_pi(p*pi<Real>());
       /*
       std::cout << std::setprecision(std::numeric_limits<Real>::max_digits10) << p << std::endl;
       std::cout << std::setprecision(std::numeric_limits<Real>::max_digits10) << Q << std::endl;
       std::cout << std::setprecision(std::numeric_limits<Real>::max_digits10) << Q_expected << std::endl;
       std::cout << fabs((Q - Q_expected) / Q_expected) << std::endl;
       */
       BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
    }
    // and for p < 1:
    for (Real p = -0.99; p < 0; p += 0.1) {
       auto f = [&](Real x)->Real
       {
          return x > 1000 * boost::math::tools::log_max_value<Real>() ? Real(0) : Real(exp(-p * log(-boost::math::expm1(-x)) - (1 + p) * x));
       };
       Q = integrator.integrate(f, get_convergence_tolerance<Real>(), &error, &L1);
       Q_expected = 1 / boost::math::sinc_pi(p*pi<Real>());
       BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
    }
}

template<class Complex>
void test_complex_modified_bessel()
{
    std::cout << "Testing complex modified Bessel function on type " << boost::typeindex::type_id<Complex>().pretty_name() << "\n";
    typedef typename Complex::value_type Real;
    Real tol = 100 * boost::math::tools::epsilon<Real>();
    Real error;
    Real L1;
    const auto& integrator = get_integrator<Real>();

    // Integral Representation of Modified Complex Bessel function:
    // https://en.wikipedia.org/wiki/Bessel_function#Modified_Bessel_functions
    Complex z{2, 3};
    const auto f = [&z](const Real& t)->Complex
    {
        using std::cosh;
        using std::exp;
        Real cosht = cosh(t);
        if (cosht > boost::math::tools::log_max_value<Real>())
        {
            return Complex{0, 0};
        }
        Complex arg = -z*cosht;
        Complex res = exp(arg);
        return res;
    };

    Complex K0 = integrator.integrate(f, get_convergence_tolerance<Real>(), &error, &L1);

    // Mathematica code: N[BesselK[0, 2 + 3 I], 140]
    #ifdef BOOST_MATH_STANDALONE
    BOOST_IF_CONSTEXPR (std::is_fundamental<Complex>::value)
    #endif
    {
        Real K0_x_expected = BOOST_MATH_TEST_VALUE(Real, -0.08296852656762551490517953520589186885781541203818846830385526187936132191822538822296497597191327722262903004145527496422090506197776994);
        Real K0_y_expected = BOOST_MATH_TEST_VALUE(Real, 0.027949603635183423629723306332336002340909030265538548521150904238352846705644065168365102147901993976999717171115546662967229050834575193041);
        BOOST_CHECK_CLOSE_FRACTION(K0.real(), K0_x_expected, tol);
        BOOST_CHECK_CLOSE_FRACTION(K0.imag(), K0_y_expected, tol);
    }
}

template<typename Complex>
void test_complex_exponential_integral_E1(){
    std::cout << "Testing complex exponential integral on type " << boost::typeindex::type_id<Complex>().pretty_name() << "\n";
    typedef typename Complex::value_type Real;
    Real tol = 100 * boost::math::tools::epsilon<Real>();
    Real error;
    Real L1;
    const auto& integrator = get_integrator<Real>();

    Complex z{1.5,0.5};

    // Integral representation of exponential integral E1, valid for Re z > 0
    // https://en.wikipedia.org/wiki/Exponential_integral#Definitions
    auto f = [&z](const Real& t)->Complex
    {
       using std::exp;
       return exp(-z*t)/t;
    };

    Real inf = std::numeric_limits<Real>::has_infinity ? std::numeric_limits<Real>::infinity() : boost::math::tools::max_value<Real>();

    Complex E1 = integrator.integrate(f,1,inf,get_convergence_tolerance<Real>(),&error,&L1);

   // Mathematica code: N[ExpIntegral[1,1.5 + 0.5 I],140]
    #ifdef BOOST_MATH_STANDALONE
    BOOST_IF_CONSTEXPR (std::is_fundamental<Complex>::value)
    #endif
    {
        Real E1_real_expected = BOOST_MATH_TEST_VALUE(Real, 0.071702995463938694845949672113596046091766639758473558841839765788732549949008866887694451956003503764943496943262401868244277788066634858393);
        Real E1_imag_expected = BOOST_MATH_TEST_VALUE(Real, -0.065138628279238400564373880665751377423524428792583839078600260273866805818117625959446311737353882269129094759883720722150048944193926087208);
        BOOST_CHECK_CLOSE_FRACTION(E1.real(), E1_real_expected, tol);
        BOOST_CHECK_CLOSE_FRACTION(E1.imag(), E1_imag_expected, tol);
    }
}

template <class T>
void test_non_central_t()
{
   //
   // Bug case from the non-central t distribution:
   //
   using std::pow;
   using std::exp;
   using std::sqrt;

   std::cout << "Testing non-central T PDF integral" << std::endl;

   T x = -1.882352352142334;
   T v = 77.384613037109375;
   T mu = 8.1538467407226562;
   T expected = static_cast<T>(4.5098555913703146875364186893655197e+49L);

   boost::math::quadrature::exp_sinh<T> integrator;
   T err;
   T L1;
   std::size_t levels;
   T integral = integrator.integrate([&x, v, mu](T y)
      {
         return pow(y, v) * exp(boost::math::pow<2>((y - mu * x / sqrt(x * x + v))) / -2);
      },
      boost::math::tools::root_epsilon<T>(), &err, &L1, &levels);

   T tol = 100 * boost::math::tools::epsilon<T>();
   BOOST_CHECK_CLOSE_FRACTION(integral, expected, tol);
}


BOOST_AUTO_TEST_CASE(exp_sinh_quadrature_test)
{
   //
   // Uncomment to generate the coefficients:
   //

   /*
   std::cout << std::scientific << std::setprecision(8);
   print_levels(generate_constants<cpp_bin_float_100, float>(8), "f");
   std::cout << std::setprecision(18);
   print_levels(generate_constants<cpp_bin_float_100, double>(8), "");
   std::cout << std::setprecision(35);
   print_levels(generate_constants<cpp_bin_float_100, cpp_bin_float_quad>(8), "L");
   */

#ifdef TEST1

#ifdef __STDCPP_FLOAT32_T__
    test_left_limit_infinite<std::float32_t>();
    test_right_limit_infinite<std::float32_t>();
    test_nr_examples<std::float32_t>();
    test_crc<std::float32_t>();
    //test_non_central_t<float32_t>();
#else
    test_left_limit_infinite<float>();
    test_right_limit_infinite<float>();
    test_nr_examples<float>();
    test_crc<float>();
    //test_non_central_t<float>();
#endif

#endif
#ifdef TEST2

#ifdef __STDCPP_FLOAT64_T__
    test_left_limit_infinite<std::float64_t>();
    test_right_limit_infinite<std::float64_t>();
    test_nr_examples<std::float64_t>();
    test_crc<std::float64_t>();
    test_non_central_t<std::float64_t>();
#else
    test_left_limit_infinite<double>();
    test_right_limit_infinite<double>();
    test_nr_examples<double>();
    test_crc<double>();
    test_non_central_t<double>();
#endif

#endif
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
#ifdef TEST3
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    test_left_limit_infinite<long double>();
    test_right_limit_infinite<long double>();
    test_nr_examples<long double>();
    test_crc<long double>();
    test_non_central_t<long double>();
#endif
#endif
#endif
#if defined(TEST4) && defined(BOOST_MATH_RUN_MP_TESTS)
    test_left_limit_infinite<cpp_bin_float_quad>();
    test_right_limit_infinite<cpp_bin_float_quad>();
    test_nr_examples<cpp_bin_float_quad>();
    test_crc<cpp_bin_float_quad>();
#endif

#if !defined(BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS) && !defined(BOOST_MATH_NO_REAL_CONCEPT_TESTS)
#ifdef TEST5
    test_left_limit_infinite<boost::math::concepts::real_concept>();
    test_right_limit_infinite<boost::math::concepts::real_concept>();
    test_nr_examples<boost::math::concepts::real_concept>();
    test_crc<boost::math::concepts::real_concept>();
    test_non_central_t<boost::math::concepts::real_concept>();
#endif
#endif
#if defined(TEST6) && defined(BOOST_MATH_RUN_MP_TESTS)
    test_left_limit_infinite<boost::multiprecision::cpp_bin_float_50>();
    test_right_limit_infinite<boost::multiprecision::cpp_bin_float_50>();
    test_nr_examples<boost::multiprecision::cpp_bin_float_50>();
    test_crc<boost::multiprecision::cpp_bin_float_50>();
#endif
#if defined(TEST7) && defined(BOOST_MATH_RUN_MP_TESTS)
    test_left_limit_infinite<boost::multiprecision::cpp_dec_float_50>();
    test_right_limit_infinite<boost::multiprecision::cpp_dec_float_50>();
    test_nr_examples<boost::multiprecision::cpp_dec_float_50>();
    //
    // This one causes stack overflows on the CI machine, but not locally,
    // assume it's due to restricted resources on the server, and <shrug> for now...
    //
#if ! BOOST_WORKAROUND(BOOST_MSVC, == 1900) && defined(BOOST_MATH_RUN_MP_TESTS)
    test_crc<boost::multiprecision::cpp_dec_float_50>();
#endif
#endif
#ifdef TEST8
    test_complex_modified_bessel<std::complex<float>>();
    test_complex_modified_bessel<std::complex<double>>();
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    test_complex_modified_bessel<std::complex<long double>>();
#endif
#ifndef BOOST_MATH_NO_MP_TESTS
    test_complex_modified_bessel<boost::multiprecision::cpp_complex_quad>();
#endif
#endif
#ifdef TEST9
    test_complex_exponential_integral_E1<std::complex<float>>();
    test_complex_exponential_integral_E1<std::complex<double>>();
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    test_complex_exponential_integral_E1<std::complex<long double>>();
#endif
#if defined(BOOST_MATH_RUN_MP_TESTS)
    test_complex_exponential_integral_E1<boost::multiprecision::cpp_complex_quad>();
#endif
#endif
#ifdef TEST10
#if defined(BOOST_HAS_FLOAT128) && !defined(BOOST_MATH_NO_MP_TESTS)
    test_complex_modified_bessel<boost::multiprecision::complex128>();
    test_complex_exponential_integral_E1<boost::multiprecision::complex128>();
#endif
#endif
}
