// Copyright Nick Thompson, 2017
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_TEST_MODULE tanh_sinh_quadrature_test

#include <boost/math/tools/config.hpp>
#include <boost/detail/workaround.hpp>

#if !defined(BOOST_NO_CXX11_DECLTYPE) && !defined(BOOST_NO_CXX11_TRAILING_RESULT_TYPES) && !defined(BOOST_NO_SFINAE_EXPR)

#include <boost/math/concepts/real_concept.hpp>
#include <boost/test/included/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/tools/test_value.hpp>
#include <boost/math/quadrature/tanh_sinh.hpp>
#include <boost/math/special_functions/sinc.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/math/special_functions/next.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/beta.hpp>
#include <boost/math/special_functions/ellint_rc.hpp>
#include <boost/math/special_functions/ellint_rj.hpp>
#include <boost/type_index.hpp>

#if __has_include(<stdfloat>)
#  include <stdfloat>
#endif

#ifdef BOOST_HAS_FLOAT128
#include <boost/multiprecision/float128.hpp>
#endif

#ifdef _MSC_VER
#pragma warning(disable:4127)  // Conditional expression is constant
#endif

#if !defined(TEST1) && !defined(TEST2) && !defined(TEST3) && !defined(TEST4) && !defined(TEST5) && !defined(TEST6) && !defined(TEST7) && !defined(TEST8)\
    && !defined(TEST1A) && !defined(TEST1B) && !defined(TEST2A) && !defined(TEST3A) && !defined(TEST6A) && !defined(TEST9)
#  define TEST1
#  define TEST2
#  define TEST3
#  define TEST4
#  define TEST5
#  define TEST6
#  define TEST7
#  define TEST8
#  define TEST1A
#  define TEST1B
#  define TEST2A
#  define TEST3A
#  define TEST6A
#  define TEST9
#endif

using std::expm1;
using std::atan;
using std::tan;
using std::log;
using std::log1p;
using std::asinh;
using std::atanh;
using std::sqrt;
using std::isnormal;
using std::abs;
using std::sinh;
using std::tanh;
using std::cosh;
using std::pow;
using std::exp;
using std::sin;
using std::cos;
using std::string;
using boost::multiprecision::cpp_bin_float_50;
using boost::multiprecision::cpp_bin_float_100;
using boost::multiprecision::cpp_dec_float_50;
using boost::multiprecision::cpp_dec_float_100;
using boost::multiprecision::cpp_bin_float_quad;
using boost::math::sinc_pi;
using boost::math::quadrature::tanh_sinh;
using boost::math::quadrature::detail::tanh_sinh_detail;
using boost::math::constants::pi;
using boost::math::constants::half_pi;
using boost::math::constants::two_div_pi;
using boost::math::constants::two_pi;
using boost::math::constants::half;
using boost::math::constants::third;
using boost::math::constants::half;
using boost::math::constants::third;
using boost::math::constants::catalan;
using boost::math::constants::ln_two;
using boost::math::constants::root_two;
using boost::math::constants::root_two_pi;
using boost::math::constants::root_pi;

template <class Real>
inline Real cast_mp_to_real(const cpp_bin_float_100& arg)
{
   return static_cast<Real>(arg);
}
template <>
inline boost::math::concepts::real_concept cast_mp_to_real<boost::math::concepts::real_concept>(const cpp_bin_float_100& arg)
{
   return static_cast<boost::math::concepts::real_concept>(static_cast<long double>(arg));
}

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
void print_complement_indexes(const T& v)
{
   std::cout << "\n   {";
   for (unsigned i = 0; i < v.size(); ++i)
   {
      unsigned index = 0;
      while (v[i][index] >= 0)
         ++index;
      std::cout << index << ", ";
   }
   std::cout << "};\n";
}

template <class T>
void print_levels(const std::pair<T, T>& p, const char* suffix = "")
{
   std::cout << "   static const std::vector<std::vector<Real> > abscissa = ";
   print_levels(p.first, suffix);
   std::cout << "   static const std::vector<std::vector<Real> > weights = ";
   print_levels(p.second, suffix);
   std::cout << "   static const std::vector<unsigned > indexes = ";
   print_complement_indexes(p.first);
}

template <class Real>
std::pair<std::vector<std::vector<Real>>, std::vector<std::vector<Real>> > generate_constants(unsigned max_index, unsigned max_rows)
{
   using boost::math::constants::half_pi;
   using boost::math::constants::two_div_pi;
   using boost::math::constants::pi;
   auto g = [](Real t) { return tanh(half_pi<Real>()*sinh(t)); };
   auto w = [](Real t) { Real cs = cosh(half_pi<Real>() * sinh(t));  return half_pi<Real>() * cosh(t) / (cs * cs); };
   auto gc = [](Real t) { Real u2 = half_pi<Real>() * sinh(t);  return 1 / (exp(u2) *cosh(u2)); };
   auto g_inv = [](float x)->float { return asinh(two_div_pi<float>()*atanh(x)); };
   auto gc_inv = [](float x)
   {
      float l = log(sqrt((2 - x) / x));
      return log((sqrt(4 * l * l + pi<float>() * pi<float>()) + 2 * l) / pi<float>());
   };

   std::vector<std::vector<Real>> abscissa, weights;

   std::vector<Real> temp;

   float t_crossover = gc_inv(0.5f);

   Real h = 1;
   for (unsigned i = 0; i < max_index; ++i)
   {
      temp.push_back((i < t_crossover) ? g(i * h) : -gc(i * h));
   }
   abscissa.push_back(temp);
   temp.clear();

   for (unsigned i = 0; i < max_index; ++i)
   {
      temp.push_back(w(i * h));
   }
   weights.push_back(temp);
   temp.clear();

   for (unsigned row = 1; row < max_rows; ++row)
   {
      h /= 2;
      for (Real t = h; t < max_index - 1; t += 2 * h)
         temp.push_back((t < t_crossover) ? g(t) : -gc(t));
      abscissa.push_back(temp);
      temp.clear();
   }
   h = 1;
   for (unsigned row = 1; row < max_rows; ++row)
   {
      h /= 2;
      for (Real t = h; t < max_index - 1; t += 2 * h)
         temp.push_back(w(t));
      weights.push_back(temp);
      temp.clear();
   }

   return std::make_pair(abscissa, weights);
}

template <class Real>
const tanh_sinh<Real>& get_integrator()
{
   static const tanh_sinh<Real> integrator(15);
   return integrator;
}

template <class Real>
Real get_convergence_tolerance()
{
   return boost::math::tools::root_epsilon<Real>();
}


template<class Real>
void test_linear()
{
    std::cout << "Testing linear functions are integrated properly by tanh_sinh on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    Real tol = 10*boost::math::tools::epsilon<Real>();
    auto integrator = get_integrator<Real>();
    auto f = [](const Real& x)
    {
       return 5*x + 7;
    };
    Real error;
    Real L1;
    Real Q = integrator.integrate(f, (Real) 0, (Real) 1, get_convergence_tolerance<Real>(), &error, &L1);
    BOOST_CHECK_CLOSE_FRACTION(Q, static_cast<Real>(9.5), tol);
    BOOST_CHECK_CLOSE_FRACTION(L1, static_cast<Real>(9.5), tol);
    Q = integrator.integrate(f, static_cast<Real>(1), static_cast<Real>(0), get_convergence_tolerance<Real>(), &error, &L1);
    BOOST_CHECK_CLOSE_FRACTION(Q, static_cast<Real>(-9.5), tol);
    BOOST_CHECK_CLOSE_FRACTION(L1, static_cast<Real>(9.5), tol);
    Q = integrator.integrate(f, static_cast<Real>(1), static_cast<Real>(1), get_convergence_tolerance<Real>(), &error, &L1);
    BOOST_CHECK_EQUAL(Q, Real(0));
}


template<class Real>
void test_quadratic()
{
    std::cout << "Testing quadratic functions are integrated properly by tanh_sinh on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    Real tol = 10*boost::math::tools::epsilon<Real>();
    auto integrator = get_integrator<Real>();
    auto f = [](const Real& x) { return 5*x*x + 7*x + 12; };
    Real error;
    Real L1;
    Real Q = integrator.integrate(f, (Real) 0, (Real) 1, get_convergence_tolerance<Real>(), &error, &L1);
    BOOST_CHECK_CLOSE_FRACTION(Q, (Real) 17 + half<Real>()*third<Real>(), tol);
    BOOST_CHECK_CLOSE_FRACTION(L1, (Real) 17 + half<Real>()*third<Real>(), tol);
}


template<class Real>
void test_singular()
{
    std::cout << "Testing integration of endpoint singularities on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    Real tol = 10*boost::math::tools::epsilon<Real>();
    Real error;
    Real L1;
    auto integrator = get_integrator<Real>();
    auto f = [](const Real& x) { return log(x)*log(1-x); };
    Real Q = integrator.integrate(f, (Real) 0, (Real) 1, get_convergence_tolerance<Real>(), &error, &L1);
    Real Q_expected = 2 - pi<Real>()*pi<Real>()*half<Real>()*third<Real>();

    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
    BOOST_CHECK_CLOSE_FRACTION(L1, Q_expected, tol);
}


// Examples taken from
//http://crd-legacy.lbl.gov/~dhbailey/dhbpapers/quadrature.pdf
template<class Real>
void test_ca()
{
    std::cout << "Testing integration of C(a) on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    Real tol = 10 * boost::math::tools::epsilon<Real>();
    Real error;
    Real L1;

    auto integrator = get_integrator<Real>();
    auto f1 = [](const Real& x) { return atan(x)/(x*(x*x + 1)) ; };
    Real Q = integrator.integrate(f1, (Real) 0, (Real) 1, get_convergence_tolerance<Real>(), &error, &L1);
    Real Q_expected = pi<Real>()*ln_two<Real>()/8 + catalan<Real>()*half<Real>();
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
    BOOST_CHECK_CLOSE_FRACTION(L1, Q_expected, tol);

    auto f2 = [](Real x)->Real { Real t0 = x*x + 1; Real t1 = sqrt(t0); return atan(t1)/(t0*t1); };
    Q = integrator.integrate(f2, (Real) 0 , (Real) 1, get_convergence_tolerance<Real>(), &error, &L1);
    Q_expected = pi<Real>()/4 - pi<Real>()/root_two<Real>() + 3*atan(root_two<Real>())/root_two<Real>();
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
    BOOST_CHECK_CLOSE_FRACTION(L1, Q_expected, tol);

    auto f5 = [](Real t)->Real { return t*t*log(t)/((t*t - 1)*(t*t*t*t + 1)); };
    Q = integrator.integrate(f5, (Real) 0 , (Real) 1);
    Q_expected = pi<Real>()*pi<Real>()*(2 - root_two<Real>())/32;
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);


    // Oh it suffers on this one.
    auto f6 = [](Real t)->Real { Real x = log(t); return x*x; };
    Q = integrator.integrate(f6, (Real) 0 , (Real) 1, get_convergence_tolerance<Real>(), &error, &L1);
    Q_expected = 2;

    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, 50*tol);
    BOOST_CHECK_CLOSE_FRACTION(L1, Q_expected, 50*tol);


    // Although it doesn't get to the requested tolerance on this integral, the error bounds can be queried and are reasonable:
    tol = sqrt(boost::math::tools::epsilon<Real>());
    auto f7 = [](const Real& t) { return sqrt(tan(t)); };
    Q = integrator.integrate(f7, (Real) 0 , (Real) half_pi<Real>(), get_convergence_tolerance<Real>(), &error, &L1);
    Q_expected = pi<Real>()/root_two<Real>();
    //
    // Slightly higher tolerance for type float, this marginal change was
    // caused by no more than changing the order in which the terms are summed:
    //
    BOOST_IF_CONSTEXPR (std::is_same<Real, float>::value 
                       #ifdef __STDCPP_FLOAT32_T__
                       || std::is_same<Real, std::float32_t>::value
                       #endif
                       )
    {
        tol *= static_cast<Real>(1.5);
    }
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
    BOOST_CHECK_CLOSE_FRACTION(L1, Q_expected, tol);

    auto f8 = [](const Real& t) { return log(cos(t)); };
    Q = integrator.integrate(f8, (Real) 0 , half_pi<Real>(), get_convergence_tolerance<Real>(), &error, &L1);
    Q_expected = -pi<Real>()*ln_two<Real>()*half<Real>();
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
    BOOST_CHECK_CLOSE_FRACTION(L1, -Q_expected, tol);
}


template<class Real>
void test_three_quadrature_schemes_examples()
{
    std::cout << "Testing integral in 'A Comparison of Three High Precision Quadrature Schemes' on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    Real tol = 10 * boost::math::tools::epsilon<Real>();
    Real Q;
    Real Q_expected;

    auto integrator = get_integrator<Real>();
    // Example 1:
    auto f1 = [](const Real& t) { return t*boost::math::log1p(t); };
    Q = integrator.integrate(f1, (Real) 0 , (Real) 1);
    Q_expected = half<Real>()*half<Real>();
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);


    // Example 2:
    auto f2 = [](const Real& t) { return t*t*atan(t); };
    Q = integrator.integrate(f2, (Real) 0 , (Real) 1);
    Q_expected = (pi<Real>() -2 + 2*ln_two<Real>())/12;
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, 2 * tol);

    // Example 3:
    auto f3 = [](const Real& t) { return exp(t)*cos(t); };
    Q = integrator.integrate(f3, (Real) 0, half_pi<Real>());
    Q_expected = boost::math::expm1(half_pi<Real>())*half<Real>();
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);

    // Example 4:
    auto f4 = [](Real x)->Real { Real t0 = sqrt(x*x + 2); return atan(t0)/(t0*(x*x+1)); };
    Q = integrator.integrate(f4, (Real) 0 , (Real) 1);
    Q_expected = 5*pi<Real>()*pi<Real>()/96;
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);

    // Example 5:
    auto f5 = [](const Real& t) { return sqrt(t)*log(t); };
    Q = integrator.integrate(f5, (Real) 0 , (Real) 1);
    Q_expected = -4/ (Real) 9;
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);

    // Example 6:
    auto f6 = [](const Real& t) { return sqrt(1 - t*t); };
    Q = integrator.integrate(f6, (Real) 0 , (Real) 1);
    Q_expected = pi<Real>()/4;
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
}


template<class Real>
void test_integration_over_real_line()
{
    std::cout << "Testing integrals over entire real line in 'A Comparison of Three High Precision Quadrature Schemes' on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    Real tol = 10 * boost::math::tools::epsilon<Real>();
    Real Q;
    Real Q_expected;
    Real error;
    Real L1;
    auto integrator = get_integrator<Real>();

    auto f1 = [](const Real& t) { return 1/(1+t*t);};
    Q = integrator.integrate(f1, std::numeric_limits<Real>::has_infinity ? -std::numeric_limits<Real>::infinity() : -boost::math::tools::max_value<Real>(), std::numeric_limits<Real>::has_infinity ? std::numeric_limits<Real>::infinity() : boost::math::tools::max_value<Real>(), get_convergence_tolerance<Real>(), &error, &L1);
    Q_expected = pi<Real>();
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
    BOOST_CHECK_CLOSE_FRACTION(L1, Q_expected, tol);

    auto f2 = [](const Real& t) { return exp(-t*t*half<Real>()); };
    Q = integrator.integrate(f2, std::numeric_limits<Real>::has_infinity ? -std::numeric_limits<Real>::infinity() : -boost::math::tools::max_value<Real>(), std::numeric_limits<Real>::has_infinity ? std::numeric_limits<Real>::infinity() : boost::math::tools::max_value<Real>(), get_convergence_tolerance<Real>(), &error, &L1);
    Q_expected = root_two_pi<Real>();
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol * 2);
    BOOST_CHECK_CLOSE_FRACTION(L1, Q_expected, tol * 2);

    // This test shows how oscillatory integrals are approximated very poorly by this method:
    //std::cout << "Testing sinc integral: \n";
    //Q = integrator.integrate(boost::math::sinc_pi<Real>, -std::numeric_limits<Real>::infinity(), std::numeric_limits<Real>::infinity(), &error, &L1);
    //std::cout << "Error estimate of sinc integral: " << error << std::endl;
    //std::cout << "L1 norm of sinc integral " << L1 << std::endl;
    //Q_expected = pi<Real>();
    //BOOST_CHECK_CLOSE(Q, Q_expected, 100*tol);

    auto f4 = [](const Real& t) { return 1/cosh(t);};
    Q = integrator.integrate(f4, std::numeric_limits<Real>::has_infinity ? -std::numeric_limits<Real>::infinity() : -boost::math::tools::max_value<Real>(), std::numeric_limits<Real>::has_infinity ? std::numeric_limits<Real>::infinity() : boost::math::tools::max_value<Real>(), get_convergence_tolerance<Real>(), &error, &L1);
    Q_expected = pi<Real>();
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
    BOOST_CHECK_CLOSE_FRACTION(L1, Q_expected, tol);

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

    // Example 11:
    auto f1 = [](const Real& t) { return 1/(1+t*t);};
    Q = integrator.integrate(f1, 0, std::numeric_limits<Real>::has_infinity ? std::numeric_limits<Real>::infinity() : boost::math::tools::max_value<Real>(), get_convergence_tolerance<Real>(), &error, &L1);
    Q_expected = half_pi<Real>();
    BOOST_CHECK_CLOSE(Q, Q_expected, 100*tol);

    // Example 12
    auto f2 = [](const Real& t) { return exp(-t)/sqrt(t); };
    Q = integrator.integrate(f2, 0, std::numeric_limits<Real>::has_infinity ? std::numeric_limits<Real>::infinity() : boost::math::tools::max_value<Real>(), get_convergence_tolerance<Real>(), &error, &L1);
    Q_expected = root_pi<Real>();
    BOOST_CHECK_CLOSE(Q, Q_expected, 1000*tol);

    auto f3 = [](const Real& t) { return exp(-t)*cos(t); };
    Q = integrator.integrate(f3, 0, std::numeric_limits<Real>::has_infinity ? std::numeric_limits<Real>::infinity() : boost::math::tools::max_value<Real>(), get_convergence_tolerance<Real>(), &error, &L1);
    Q_expected = half<Real>();
    BOOST_CHECK_CLOSE(Q, Q_expected, 100*tol);

    auto f4 = [](const Real& t) { return 1/(1+t*t); };
    Q = integrator.integrate(f4, 1, std::numeric_limits<Real>::has_infinity ? std::numeric_limits<Real>::infinity() : boost::math::tools::max_value<Real>(), get_convergence_tolerance<Real>(), &error, &L1);
    Q_expected = pi<Real>()/4;
    BOOST_CHECK_CLOSE(Q, Q_expected, 100*tol);
}

template<class Real>
void test_left_limit_infinite()
{
    std::cout << "Testing left limit infinite for tanh_sinh in 'A Comparison of Three High Precision Quadrature Schemes' on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    Real tol = 10 * boost::math::tools::epsilon<Real>();
    Real Q;
    Real Q_expected;
    const auto& integrator = get_integrator<Real>();

    // Example 11:
    auto f1 = [](const Real& t) { return 1/(1+t*t);};
    Q = integrator.integrate(f1, std::numeric_limits<Real>::has_infinity ? -std::numeric_limits<Real>::infinity() : -boost::math::tools::max_value<Real>(), Real(0));
    Q_expected = half_pi<Real>();
    BOOST_CHECK_CLOSE(Q, Q_expected, 100*tol);
}


// A horrible function taken from
// http://www.chebfun.org/examples/quad/GaussClenCurt.html
template<class Real>
void test_horrible()
{
   #ifdef BOOST_MATH_STANDALONE
   BOOST_IF_CONSTEXPR (std::is_fundamental<Real>::value)
   #endif
   {
      std::cout << "Testing Trefenthen's horrible integral on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
      // We only know the integral to double precision, so requesting a higher tolerance doesn't make sense.
      Real tol = 10 * std::numeric_limits<float>::epsilon();
      Real Q;
      Real Q_expected;
      Real error;
      Real L1;
      const auto& integrator = get_integrator<Real>();

      auto f = [](Real x)->Real { return x*sin(2*exp(2*sin(2*exp(2*x) ) ) ); };
      Q = integrator.integrate(f, (Real) -1, (Real) 1, get_convergence_tolerance<Real>(), &error, &L1);
      // NIntegrate[x*Sin[2*Exp[2*Sin[2*Exp[2*x]]]], {x, -1, 1}, WorkingPrecision -> 130, MaxRecursion -> 100]
      Q_expected = BOOST_MATH_TEST_VALUE(Real, 0.33673283478172753598559003181355241139806404130031017259552729882281);
      BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
      // Over again without specifying the bounds:
      Q = integrator.integrate(f, get_convergence_tolerance<Real>(), &error, &L1);
      BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
   }
}

// Some examples of tough integrals from NR, section 4.5.4:
template<class Real>
void test_nr_examples()
{
    std::cout << "Testing singular integrals from NR 4.5.4 on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    Real tol = 10 * boost::math::tools::epsilon<Real>();
    Real Q;
    Real Q_expected;
    Real error;
    Real L1;
    const auto& integrator = get_integrator<Real>();

    auto f1 = [](Real x)->Real
    {
       return (sin(x * half<Real>()) * exp(-x) / x) / sqrt(x);
    };
    Q = integrator.integrate(f1, 0, std::numeric_limits<Real>::has_infinity ? std::numeric_limits<Real>::infinity() : boost::math::tools::max_value<Real>(), get_convergence_tolerance<Real>(), &error, &L1);
    Q_expected = sqrt(pi<Real>()*(sqrt((Real) 5) - 2));
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, 25*tol);

    auto f2 = [](Real x)->Real { return pow(x, -(Real) 2/(Real) 7)*exp(-x*x); };
    Q = integrator.integrate(f2, 0, std::numeric_limits<Real>::has_infinity ? std::numeric_limits<Real>::infinity() : boost::math::tools::max_value<Real>());
    Q_expected = half<Real>()*boost::math::tgamma((Real) 5/ (Real) 14);
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol * 10);

}

// Test integrand known to fool some termination schemes:
template<class Real>
void test_early_termination()
{
    std::cout << "Testing Clenshaw & Curtis's example of integrand which fools termination schemes on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    Real tol = 10 * boost::math::tools::epsilon<Real>();
    Real Q;
    Real Q_expected;
    Real error;
    Real L1;
    const auto& integrator = get_integrator<Real>();

    auto f1 = [](Real x)->Real { return 23*cosh(x)/ (Real) 25 - cos(x) ; };
    Q = integrator.integrate(f1, (Real) -1, (Real) 1, get_convergence_tolerance<Real>(), &error, &L1);
    Q_expected = 46*sinh((Real) 1)/(Real) 25 - 2*sin((Real) 1);
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
    // Over again with no bounds:
    Q = integrator.integrate(f1);
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
}

// Test some definite integrals from the CRC handbook, 32nd edition:
template<class Real>
void test_crc()
{
    std::cout << "Testing CRC formulas on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    Real tol = 10 * boost::math::tools::epsilon<Real>();
    Real Q;
    Real Q_expected;
    Real error;
    Real L1;
    const auto& integrator = get_integrator<Real>();

    // CRC Definite integral 585
    auto f1 = [](Real x)->Real { Real t = log(1/x); return x*x*t*t*t; };
    Q = integrator.integrate(f1, (Real) 0, (Real) 1, get_convergence_tolerance<Real>(), &error, &L1);
    Q_expected = (Real) 2/ (Real) 27;
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);

    // CRC 636:
    #ifdef BOOST_MATH_STANDALONE
    BOOST_IF_CONSTEXPR (std::is_fundamental<Real>::value)
    #endif
    {
      auto f2 = [](Real x)->Real { return sqrt(cos(x)); };
      Q = integrator.integrate(f2, (Real) 0, (Real) half_pi<Real>(), get_convergence_tolerance<Real>(), &error, &L1);
      //Q_expected = pow(two_pi<Real>(), 3*half<Real>())/pow(boost::math::tgamma((Real) 1/ (Real) 4), 2);
      Q_expected = BOOST_MATH_TEST_VALUE(Real, 1.1981402347355922074399224922803238782272126632156515582636749529);
      BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
    

      // CRC Section 5.5, integral 585:
      for (int n = 0; n < 3; ++n) {
         for (int m = 0; m < 3; ++m) {
               auto f = [&](Real x)->Real { return pow(x, Real(m))*pow(log(1/x), Real(n)); };
               Q = integrator.integrate(f, (Real) 0, (Real) 1, get_convergence_tolerance<Real>(), &error, &L1);
               // Calculation of the tgamma function is not exact, giving spurious failures.
               // Casting to cpp_bin_float_100 beforehand fixes most of them.
               cpp_bin_float_100 np1 = n + 1;
               cpp_bin_float_100 mp1 = m + 1;
               Q_expected = cast_mp_to_real<Real>(tgamma(np1)/pow(mp1, np1));
               BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
         }
      }
    }
    // CRC Section 5.5, integral 591
    // The parameter p allows us to control the strength of the singularity.
    // Rapid convergence is not guaranteed for this function, as the branch cut makes it non-analytic on a disk.
    // This converges only when our test type has an extended exponent range as all the area of the integral
    // occurs so close to 0 (or 1) that we need abscissa values exceptionally small to find it.
    // "There's a lot of room at the bottom".
    // We also use a 2 argument functor so that 1-x is evaluated accurately:
    if (std::numeric_limits<Real>::max_exponent > std::numeric_limits<double>::max_exponent)
    {
       for (Real p = Real (-0.99); p < 1; p += Real(0.1)) {
          auto f = [&](Real x, Real cx)->Real
          {
             //return pow(x, p) / pow(1 - x, p);
             return cx < 0 ? exp(p * (log(x) - boost::math::log1p(-x))) : pow(x, p) / pow(cx, p);
          };
          Q = integrator.integrate(f, (Real)0, (Real)1, get_convergence_tolerance<Real>(), &error, &L1);
          Q_expected = 1 / sinc_pi(p*pi<Real>());
          BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, 10 * tol);
       }
    }
    // There is an alternative way to evaluate the above integral: by noticing that all the area of the integral
    // is near zero for p < 0 and near 1 for p > 0 we can substitute exp(-x) for x and remap the integral to the
    // domain (0, INF).  Internally we need to expand out the terms and evaluate using logs to avoid spurious overflow, 
    // this gives us
    // for p > 0:
    for (Real p = Real(0.99); p > 0; p -= Real(0.1)) {
       auto f = [&](Real x)->Real
       {
          return exp(-x * (1 - p) + p * log(-boost::math::expm1(-x)));
       };
       Q = integrator.integrate(f, 0, boost::math::tools::max_value<Real>(), get_convergence_tolerance<Real>(), &error, &L1);
       Q_expected = 1 / sinc_pi(p*pi<Real>());
       BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, 10 * tol);
    }
    // and for p < 1:
    for (Real p = Real (-0.99); p < 0; p += Real(0.1)) {
       auto f = [&](Real x)->Real
       {
          return exp(-p * log(-boost::math::expm1(-x)) - (1 + p) * x);
       };
       Q = integrator.integrate(f, 0, boost::math::tools::max_value<Real>(), get_convergence_tolerance<Real>(), &error, &L1);
       Q_expected = 1 / sinc_pi(p*pi<Real>());
       BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, 10 * tol);
    }

    // CRC Section 5.5, integral 635
    for (int m = 0; m < 10; ++m) {
        auto f = [&](Real x)->Real { return Real(1)/(Real(1) + pow(tan(x), m)); };
        Q = integrator.integrate(f, (Real) 0, half_pi<Real>(), get_convergence_tolerance<Real>(), &error, &L1);
        Q_expected = half_pi<Real>()/2;
        BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
    }

    // CRC Section 5.5, integral 637:
    //
    // When h gets very close to 1, the strength of the singularity gradually increases until we
    // no longer have enough exponent range to evaluate the integral....
    // We also have to use the 2-argument functor version of the integrator to avoid
    // cancellation error, since the singularity is near PI/2.
    //
    Real limit = std::numeric_limits<Real>::max_exponent > std::numeric_limits<double>::max_exponent
       ? .95f : .85f;
    for (Real h = Real(0.01); h < limit; h += Real(0.1)) {
        auto f = [&](Real x, Real xc)->Real { return xc > 0 ? pow(1/tan(xc), h) : pow(tan(x), h); };
        Q = integrator.integrate(f, (Real) 0, half_pi<Real>(), get_convergence_tolerance<Real>(), &error, &L1);
        Q_expected = half_pi<Real>()/cos(h*half_pi<Real>());
        BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
    }
    // CRC Section 5.5, integral 637:
    //
    // Over again, but with argument transformation, we want:
    //
    // Integral of tan(x)^h over (0, PI/2)
    //
    // Note that the bulk of the area is next to the singularity at PI/2,
    // so we'll start by replacing x by PI/2 - x, and that tan(PI/2 - x) == 1/tan(x)
    // so we now have:
    //
    // Integral of 1/tan(x)^h over (0, PI/2)
    //
    // Which is almost the ideal form, except that when h is very close to 1
    // we run out of exponent range in evaluating the integral arbitrarily close to 0.
    // So now we substitute exp(-x) for x: this stretches out the range of the
    // integral to (-log(PI/2), +INF) with the singularity at +INF giving:
    //
    // Integral of exp(-x)/tan(exp(-x))^h over (-log(PI/2), +INF)
    //
    // We just need a way to evaluate the function without spurious under/overflow
    // in the exp terms.  Note that for small x: tan(x) ~= x, so making this
    // substitution and evaluating by logs we have:
    //
    // exp(-x)/tan(exp(-x))^h ~= exp((h - 1) * x)  for x > -log(epsilon);
    //
    // Here's how that looks in code:
    //
    for (Real i = 80; i < 100; ++i) {
       Real h = i / 100;
       auto f = [&](Real x)->Real 
       { 
          if (x > -log(boost::math::tools::epsilon<Real>()))
             return exp((h - 1) * x);
          else
          {
             Real et = exp(-x);
             // Need to deal with numeric instability causing et to be greater than PI/2:
             return et >= boost::math::constants::half_pi<Real>() ? 0 : et * pow(1 / tan(et), h);
          }
       };
       Q = integrator.integrate(f, -log(half_pi<Real>()), boost::math::tools::max_value<Real>(), get_convergence_tolerance<Real>(), &error, &L1);
       Q_expected = half_pi<Real>() / cos(h*half_pi<Real>());
       BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, 5 * tol);
    }

    // CRC Section 5.5, integral 670:

    auto f3 = [](Real x)->Real { return sqrt(log(1/x)); };
    Q = integrator.integrate(f3, (Real) 0, (Real) 1, get_convergence_tolerance<Real>(), &error, &L1);
    Q_expected = root_pi<Real>()/2;
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);

}

template <class Real>
void test_sf()
{
   using std::sqrt;
   // Test some special functions that we already know how to evaluate:
   std::cout << "Testing special functions on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
   Real tol = 10 * boost::math::tools::epsilon<Real>();
   const auto& integrator = get_integrator<Real>();

   // incomplete beta:
   if (std::numeric_limits<Real>::digits10 < 37) // Otherwise too slow
   {
      Real a(100), b(15);
      auto f = [&](Real x)->Real { return boost::math::ibeta_derivative(a, b, x); };
      BOOST_CHECK_CLOSE_FRACTION(integrator.integrate(f, 0, Real(0.25)), boost::math::ibeta(100, 15, Real(0.25)), tol * 10);
      // Check some really extreme versions:
      a = 1000;
      b = 500;
      BOOST_CHECK_CLOSE_FRACTION(integrator.integrate(f, 0, 1), Real(1), tol * 15);
      //
      // This is as extreme as we can get in this domain: otherwise the function has all it's 
      // area so close to zero we never get in there no matter how many levels we go down:
      //
      a = Real(1) / 15;
      b = 30;
      BOOST_CHECK_CLOSE_FRACTION(integrator.integrate(f, 0, 1), Real(1), tol * 25);
   }

   Real x = 2, y = 3, z = 0.5, p = 0.25;
   // Elliptic integral RC:
   Real Q = integrator.integrate([&](const Real& t)->Real { return 1 / (sqrt(t + x) * (t + y)); }, 0, std::numeric_limits<Real>::has_infinity ? std::numeric_limits<Real>::infinity() : boost::math::tools::max_value<Real>()) / 2;
   BOOST_CHECK_CLOSE_FRACTION(Q, boost::math::ellint_rc(x, y), tol);
   // Elliptic Integral RJ:
   BOOST_CHECK_CLOSE_FRACTION(Real((Real(3) / 2) * integrator.integrate([&](Real t)->Real { return 1 / (sqrt((t + x) * (t + y) * (t + z)) * (t + p)); }, 0, std::numeric_limits<Real>::has_infinity ? std::numeric_limits<Real>::infinity() : boost::math::tools::max_value<Real>())), boost::math::ellint_rj(x, y, z, p), tol);

   z = 5.5;
   if (std::numeric_limits<Real>::digits10 > 40)
      tol *= 200;
   else if (!std::numeric_limits<Real>::is_specialized)
      tol *= 3;
   // tgamma expressed as an integral:
   BOOST_CHECK_CLOSE_FRACTION(integrator.integrate([&](Real t)->Real { using std::pow; using std::exp; return t > 10000 ? Real(0) : Real(pow(t, z - 1) * exp(-t)); }, 0, std::numeric_limits<Real>::has_infinity ? std::numeric_limits<Real>::infinity() : boost::math::tools::max_value<Real>()),
      boost::math::tgamma(z), tol);
   BOOST_CHECK_CLOSE_FRACTION(integrator.integrate([](const Real& t)->Real {  using std::exp; return exp(-t*t); }, std::numeric_limits<Real>::has_infinity ? -std::numeric_limits<Real>::infinity() : -boost::math::tools::max_value<Real>(), std::numeric_limits<Real>::has_infinity ? std::numeric_limits<Real>::infinity() : boost::math::tools::max_value<Real>()),
      boost::math::constants::root_pi<Real>(), tol);
}

template <class Real>
void test_2_arg()
{
   BOOST_MATH_STD_USING
   std::cout << "Testing 2 argument functors on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
   Real tol = 10 * boost::math::tools::epsilon<Real>();

   const auto& integrator = get_integrator<Real>();

   //
   // There are a whole family of integrals of the general form
   // x(1-x)^-N ; N < 1
   // which have all the interesting behaviour near the 2 singularities
   // and all converge, see: http://www.wolframalpha.com/input/?i=integrate+(x+*+(1-x))%5E-1%2FN+from+0+to+1
   //
   Real Q = integrator.integrate([&](const Real& t, const Real & tc)->Real
   {
      return tc < 0 ? 1 / sqrt(t * (1-t)) : 1 / sqrt(t * tc);
   }, 0, 1);
   BOOST_CHECK_CLOSE_FRACTION(Q, boost::math::constants::pi<Real>(), tol);
   Q = integrator.integrate([&](const Real& t, const Real & tc)->Real
   {
      return tc < 0 ? 1 / boost::math::cbrt(t * (1-t)) : 1 / boost::math::cbrt(t * tc);
   }, 0, 1);
   BOOST_CHECK_CLOSE_FRACTION(Q, boost::math::pow<2>(boost::math::tgamma(Real(2) / 3)) / boost::math::tgamma(Real(4) / 3), tol * 20);
   //
   // We can do the same thing with ((1+x)(1-x))^-N ; N < 1
   //
   Q = integrator.integrate([&](const Real& t, const Real & tc)->Real
   {
      return t < 0 ? 1 / sqrt(-tc * (1-t)) : 1 / sqrt((t + 1) * tc);
   }, -1, 1);
   BOOST_CHECK_CLOSE_FRACTION(Q, boost::math::constants::pi<Real>(), tol);
   Q = integrator.integrate([&](const Real& t, const Real & tc)->Real
   {
      return t < 0 ? 1 / sqrt(-tc * (1-t)) : 1 / sqrt((t + 1) * tc);
   });
   BOOST_CHECK_CLOSE_FRACTION(Q, boost::math::constants::pi<Real>(), tol);
   Q = integrator.integrate([&](const Real& t, const Real & tc)->Real
   {
      return t < 0 ? 1 / boost::math::cbrt(-tc * (1-t)) : 1 / boost::math::cbrt((t + 1) * tc);
   }, -1, 1);
   BOOST_CHECK_CLOSE_FRACTION(Q, sqrt(boost::math::constants::pi<Real>()) * boost::math::tgamma(Real(2) / 3) / boost::math::tgamma(Real(7) / 6), tol * 10);
   Q = integrator.integrate([&](const Real& t, const Real & tc)->Real
   {
      return t < 0 ? 1 / boost::math::cbrt(-tc * (1-t)) : 1 / boost::math::cbrt((t + 1) * tc);
   });
   BOOST_CHECK_CLOSE_FRACTION(Q, sqrt(boost::math::constants::pi<Real>()) * boost::math::tgamma(Real(2) / 3) / boost::math::tgamma(Real(7) / 6), tol * 10);
   //
   // These are taken from above, and do not get to full precision via the single arg functor:
   //
   auto f7 = [](const Real& t, const Real& tc) { return t < 1 ? sqrt(tan(t)) : sqrt(1/tan(tc)); };
   Real error, L1;
   Q = integrator.integrate(f7, (Real)0, (Real)half_pi<Real>(), get_convergence_tolerance<Real>(), &error, &L1);
   Real Q_expected = pi<Real>() / root_two<Real>();
   BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
   BOOST_CHECK_CLOSE_FRACTION(L1, Q_expected, tol);

   auto f8 = [](const Real& t, const Real& tc) { return t < 1 ? log(cos(t)) : log(sin(tc)); };
   Q = integrator.integrate(f8, (Real)0, half_pi<Real>(), get_convergence_tolerance<Real>(), &error, &L1);
   Q_expected = -pi<Real>()*ln_two<Real>()*half<Real>();
   BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
   BOOST_CHECK_CLOSE_FRACTION(L1, -Q_expected, tol);
}

template <class Complex>
void test_complex()
{
   typedef typename Complex::value_type value_type;
   //
   // Integral version of the confluent hypergeometric function:
   // https://dlmf.nist.gov/13.4#E1
   //
   value_type tol = 10 * boost::math::tools::epsilon<value_type>();
   Complex a(2, 3), b(3, 4), z(0.5, -2);

   auto f = [&](value_type t)
   {
      return exp(z * t) * pow(t, a - value_type(1)) * pow(value_type(1) - t, b - a - value_type(1));
   };

   const auto& integrator = get_integrator<value_type>();
   auto Q = integrator.integrate(f, value_type(0), value_type(1), get_convergence_tolerance<value_type>());
   //
   // Expected result computed from http://www.wolframalpha.com/input/?i=1F1%5B(2%2B3i),+(3%2B4i);+(0.5-2i)%5D+*+gamma(2%2B3i)+*+gamma(1%2Bi)+%2F+gamma(3%2B4i)
   //
   #ifdef BOOST_MATH_STANDALONE
   BOOST_IF_CONSTEXPR (std::is_fundamental<Complex>::value)
   #endif
   {
      Complex expected(BOOST_MATH_TEST_VALUE(value_type, - 0.2911081612888249710582867318081776512805281815037891183828405999609246645054069649838607112484426042883371996),
         BOOST_MATH_TEST_VALUE(value_type, 0.4507983563969959578849120188097153649211346293694903758252662015991543519595834937475296809912196906074655385));

      value_type error = abs(expected - Q);
      BOOST_CHECK_LE(error, tol);

      //
      // Sin Integral https://dlmf.nist.gov/6.2#E9
      //
      auto f2 = [z](value_type t)
      {
         return -exp(-z * cos(t)) * cos(z * sin(t));
      };
      Q = integrator.integrate(f2, value_type(0), boost::math::constants::half_pi<value_type>(), get_convergence_tolerance<value_type>());

      expected = Complex(BOOST_MATH_TEST_VALUE(value_type, 0.8893822921008980697856313681734926564752476188106405688951257340480164694708337246829840859633322683740376134733),
         -BOOST_MATH_TEST_VALUE(value_type, 2.381380802906111364088958767973164614925936185337231718483495612539455538280372745733208000514737758457795502168));
      expected -= boost::math::constants::half_pi<value_type>();

      error = abs(expected - Q);
      BOOST_CHECK_LE(error, tol);
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

   T left = 0;
   T right = std::numeric_limits<T>::has_infinity ? std::numeric_limits<T>::infinity() : boost::math::tools::max_value<T>();

   boost::math::quadrature::tanh_sinh<T> integrator;
   T err;
   T L1;
   std::size_t levels;
   T integral = integrator.integrate([&x, v, mu](T y)
      {
         return pow(y, v) * exp(boost::math::pow<2>((y - mu * x / sqrt(x * x + v))) / -2);
      },
      left, right,
      boost::math::tools::root_epsilon<T>(), &err, &L1, &levels);

   T tol = 100 * boost::math::tools::epsilon<T>();
   BOOST_CHECK_CLOSE_FRACTION(integral, expected, tol);
}


BOOST_AUTO_TEST_CASE(tanh_sinh_quadrature_test)
{
#ifdef GENERATE_CONSTANTS
   //
   // Generate pre-computed coefficients:
   std::cout << std::setprecision(35);
   print_levels(generate_constants<cpp_bin_float_100>(10, 8), "L");

#else

#ifdef TEST1

    test_right_limit_infinite<float>();
    test_left_limit_infinite<float>();
    test_linear<float>();
    test_quadratic<float>();
    test_singular<float>();
    test_ca<float>();
    test_three_quadrature_schemes_examples<float>();
    test_horrible<float>();
    test_integration_over_real_line<float>();
    test_nr_examples<float>();

    #ifdef __STDCPP_FLOAT32_T__
    test_right_limit_infinite<std::float32_t>();
    test_left_limit_infinite<std::float32_t>();
    test_linear<std::float32_t>();
    test_quadratic<std::float32_t>();
    test_singular<std::float32_t>();
    test_ca<std::float32_t>();
    test_three_quadrature_schemes_examples<std::float32_t>();
    test_horrible<std::float32_t>();
    test_integration_over_real_line<std::float32_t>();
    test_nr_examples<std::float32_t>();
    #endif

#endif
#ifdef TEST1A
    test_early_termination<float>();
    test_2_arg<float>();
#endif
#ifdef TEST1B
    #ifndef BOOST_MATH_STANDALONE
    test_crc<float>();
    #endif
#endif
#ifdef TEST2
    test_right_limit_infinite<double>();
    test_left_limit_infinite<double>();
    test_linear<double>();
    test_quadratic<double>();
    test_singular<double>();
    test_ca<double>();
    test_three_quadrature_schemes_examples<double>();
    test_horrible<double>();
    test_integration_over_real_line<double>();
    test_nr_examples<double>();
    test_early_termination<double>();
    test_sf<double>();
    test_2_arg<double>();
    test_non_central_t<double>();
#endif
#ifdef TEST2A
   #ifndef BOOST_MATH_STANDALONE
    test_crc<double>();
   #endif
#endif

#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS

#ifdef TEST3
    test_right_limit_infinite<long double>();
    test_left_limit_infinite<long double>();
    test_linear<long double>();
    test_quadratic<long double>();
    test_singular<long double>();
    test_ca<long double>();
    test_three_quadrature_schemes_examples<long double>();
    test_horrible<long double>();
    test_integration_over_real_line<long double>();
    test_nr_examples<long double>();
    test_early_termination<long double>();
    test_sf<long double>();
    test_2_arg<long double>();
    test_non_central_t<long double>();
#endif
#ifdef TEST3A
   #ifndef BOOST_MATH_STANDALONE
    test_crc<long double>();
   #endif

#endif
#endif

#ifdef TEST4
   #ifdef BOOST_MATH_RUN_MP_TESTS
    test_right_limit_infinite<cpp_bin_float_quad>();
    test_left_limit_infinite<cpp_bin_float_quad>();
    test_linear<cpp_bin_float_quad>();
    test_quadratic<cpp_bin_float_quad>();
    test_singular<cpp_bin_float_quad>();
    test_ca<cpp_bin_float_quad>();
    test_three_quadrature_schemes_examples<cpp_bin_float_quad>();
    test_horrible<cpp_bin_float_quad>();
    test_nr_examples<cpp_bin_float_quad>();
    test_early_termination<cpp_bin_float_quad>();
    test_crc<cpp_bin_float_quad>();
    test_sf<cpp_bin_float_quad>();
    test_2_arg<cpp_bin_float_quad>();
#endif
#endif
#ifdef TEST5
   #ifdef BOOST_MATH_RUN_MP_TESTS
    test_sf<cpp_bin_float_50>();
    test_sf<cpp_bin_float_100>();
    test_sf<boost::multiprecision::number<boost::multiprecision::cpp_bin_float<150> > >();
   #endif

#endif
#if !defined(BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS) && !defined(BOOST_MATH_NO_REAL_CONCEPT_TESTS)
#ifdef TEST6

    test_right_limit_infinite<boost::math::concepts::real_concept>();
    test_left_limit_infinite<boost::math::concepts::real_concept>();
    test_linear<boost::math::concepts::real_concept>();
    test_quadratic<boost::math::concepts::real_concept>();
    test_singular<boost::math::concepts::real_concept>();
    test_ca<boost::math::concepts::real_concept>();
    test_three_quadrature_schemes_examples<boost::math::concepts::real_concept>();
    test_horrible<boost::math::concepts::real_concept>();
    test_integration_over_real_line<boost::math::concepts::real_concept>();
    test_nr_examples<boost::math::concepts::real_concept>();
    test_early_termination<boost::math::concepts::real_concept>();
    test_sf<boost::math::concepts::real_concept>();
    test_2_arg<boost::math::concepts::real_concept>();
    test_non_central_t<boost::math::concepts::real_concept>();
#endif
#ifdef TEST6A
    test_crc<boost::math::concepts::real_concept>();

#endif
#endif
#ifdef TEST7
   #ifdef BOOST_MATH_RUN_MP_TESTS
    test_sf<cpp_dec_float_50>();
   #endif
#endif
#if defined(TEST8) && defined(BOOST_HAS_FLOAT128) && !defined(BOOST_MATH_NO_MP_TESTS)

    test_right_limit_infinite<boost::multiprecision::float128>();
    test_left_limit_infinite<boost::multiprecision::float128>();
    test_linear<boost::multiprecision::float128>();
    test_quadratic<boost::multiprecision::float128>();
    test_singular<boost::multiprecision::float128>();
    test_ca<boost::multiprecision::float128>();
    test_three_quadrature_schemes_examples<boost::multiprecision::float128>();
    test_horrible<boost::multiprecision::float128>();
    test_integration_over_real_line<boost::multiprecision::float128>();
    test_nr_examples<boost::multiprecision::float128>();
    test_early_termination<boost::multiprecision::float128>();
    test_crc<boost::multiprecision::float128>();
    test_sf<boost::multiprecision::float128>();
    test_2_arg<boost::multiprecision::float128>();

#endif
#ifdef TEST9
    test_complex<std::complex<double> >();
    test_complex<std::complex<float> >();
#endif


#endif
}

#else

int main() { return 0; }

#endif
