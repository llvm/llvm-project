// Copyright Nick Thompson, 2017
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_TEST_MODULE sinh_sinh_quadrature_test
#include <complex>
#include <boost/multiprecision/cpp_complex.hpp>
#include <boost/math/concepts/real_concept.hpp>
#include <boost/test/included/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/quadrature/sinh_sinh.hpp>
#include <boost/math/special_functions/sinc.hpp>
#include <boost/math/tools/test_value.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/type_index.hpp>

#if !BOOST_WORKAROUND(BOOST_MSVC, < 1900)
// MSVC-12 has problems if we include 2 different multiprecision types in the same program,
// basically random things stop compiling, even though they work fine in isolation :(
#include <boost/multiprecision/cpp_dec_float.hpp>
#endif

using std::expm1;
using std::exp;
using std::sin;
using std::cos;
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
using std::string;
using boost::multiprecision::cpp_bin_float_quad;
using boost::math::quadrature::sinh_sinh;
using boost::math::constants::pi;
using boost::math::constants::pi_sqr;
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
//
// Code for generating the coefficients:
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
   auto g = [](Real t) { return sinh(half_pi<Real>()*sinh(t)); };
   auto w = [](Real t) { return cosh(t)*half_pi<Real>()*cosh(half_pi<Real>()*sinh(t)); };

   std::vector<std::vector<Real>> abscissa, weights;

   std::vector<Real> temp;

   Real t_max = log(2 * two_div_pi<Real>()*log(2 * two_div_pi<Real>()*sqrt(boost::math::tools::max_value<TargetType>())));

   std::cout << "m_t_max = " << t_max << ";\n";

   Real h = 1;
   for (Real t = 1; t < t_max; t += h)
   {
      temp.push_back(g(t));
   }
   abscissa.push_back(temp);
   temp.clear();

   for (Real t = 1; t < t_max; t += h)
   {
      temp.push_back(w(t * h));
   }
   weights.push_back(temp);
   temp.clear();

   for (unsigned row = 1; row < max_rows; ++row)
   {
      h /= 2;
      for (Real t = h; t < t_max; t += 2 * h)
         temp.push_back(g(t));
      abscissa.push_back(temp);
      temp.clear();
   }
   h = 1;
   for (unsigned row = 1; row < max_rows; ++row)
   {
      h /= 2;
      for (Real t = h; t < t_max; t += 2 * h)
         temp.push_back(w(t));
      weights.push_back(temp);
      temp.clear();
   }

   return std::make_pair(abscissa, weights);
}

template<class Real>
void test_nr_examples()
{
    std::cout << "Testing type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    Real integration_limit = sqrt(boost::math::tools::epsilon<Real>());
    Real tol = 10 * boost::math::tools::epsilon<Real>();
    std::cout << std::setprecision(std::numeric_limits<Real>::digits10);
    Real Q;
    Real Q_expected;
    Real L1;
    Real error;
    const sinh_sinh<Real> integrator(10);

    auto f0 = [](Real)->Real { return (Real) 0; };
    Q = integrator.integrate(f0, integration_limit, &error, &L1);
    Q_expected = 0;
    BOOST_CHECK_SMALL(Q, tol);
    BOOST_CHECK_SMALL(L1, tol);

    // In spite of the poles at \pm i, we still get a doubling of the correct digits at each level of refinement.
    auto f1 = [](const Real& t)->Real { return 1/(1+t*t); };
    Q = integrator.integrate(f1, integration_limit, &error, &L1);
    Q_expected = pi<Real>();
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
    BOOST_CHECK_CLOSE_FRACTION(L1, Q_expected, tol);
#if defined(BOOST_MSVC) && (BOOST_MSVC < 1900)
    auto f2 = [](const Real& x)->Real { return fabs(x) > boost::math::tools::log_max_value<Real>() ? 0 : exp(-x*x); };
#else
    auto f2 = [](const Real& x)->Real { return exp(-x*x); };
#endif
    Q = integrator.integrate(f2, integration_limit, &error, &L1);
    Q_expected = root_pi<Real>();
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
    BOOST_CHECK_CLOSE_FRACTION(L1, Q_expected, tol);

    auto f5 = [](const Real& t)->Real { return 1/cosh(t);};
    Q = integrator.integrate(f5, integration_limit, &error, &L1);
    Q_expected = pi<Real>();
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
    BOOST_CHECK_CLOSE_FRACTION(L1, Q_expected, tol);

    // This oscillatory integral has rapid convergence because the oscillations get swamped by the exponential growth of the denominator,
    // none the less the error is slightly higher than for the other cases:
    tol *= 10;
    auto f8 = [](const Real& t)->Real { return cos(t)/cosh(t);};
    Q = integrator.integrate(f8, integration_limit, &error, &L1);
    Q_expected = pi<Real>()/cosh(half_pi<Real>());
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
    // Try again with progressively fewer arguments:
    Q = integrator.integrate(f8, integration_limit);
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
    Q = integrator.integrate(f8);
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
}

// Test formulas for in the CRC Handbook of Mathematical functions, 32nd edition.
template<class Real>
void test_crc()
{
    std::cout << "Testing CRC formulas on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    Real integration_limit = sqrt(boost::math::tools::epsilon<Real>());
    Real tol = 10 * boost::math::tools::epsilon<Real>();
    std::cout << std::setprecision(std::numeric_limits<Real>::digits10);
    Real Q;
    Real Q_expected;
    Real L1;
    Real error;
    const sinh_sinh<Real> integrator(10);

    // CRC Definite integral 698:
    auto f0 = [](Real x)->Real {
      using std::sinh;
      if(x == 0) {
        return (Real) 1;
      }
      return x/sinh(x);
    };
    Q = integrator.integrate(f0, integration_limit, &error, &L1);
    Q_expected = pi_sqr<Real>()/2;
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
    BOOST_CHECK_CLOSE_FRACTION(L1, Q_expected, tol);


    // CRC Definite integral 695:
    auto f1 = [](Real x)->Real {
      using std::sin; using std::sinh;
      if(x == 0) {
        return (Real) 1;
      }
      return (Real) sin(x)/sinh(x);
    };
    Q = integrator.integrate(f1, integration_limit, &error, &L1);
    Q_expected = pi<Real>()*tanh(half_pi<Real>());
    BOOST_CHECK_CLOSE_FRACTION(Q, Q_expected, tol);
}

template<class Complex>
void test_dirichlet_eta()
{
  typedef typename Complex::value_type Real;
  std::cout << "Testing Dirichlet eta function on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
  Real tol = 10 * boost::math::tools::epsilon<Real>();
  Complex Q;
  const sinh_sinh<Real> integrator(10);

  //https://en.wikipedia.org/wiki/Dirichlet_eta_function, integral representations:
  Complex z = {1,1};
  auto eta = [&z](Real t)->Complex {
    using std::exp;
    using std::pow;
    using boost::math::constants::pi;
    Complex i = {0,1};
    Complex num = pow((Real)1/ (Real)2  + i*t, -z);
    Real denom = exp(pi<Real>()*t) + exp(-pi<Real>()*t);
    Complex res = num/denom;
    return res;
  };
  Q = integrator.integrate(eta);
  // N[DirichletEta[1 + I], 150]
  {
    Complex Q_expected = { BOOST_MATH_TEST_VALUE(Real, 0.7265597750624632632014957285472413863111295027357257871), BOOST_MATH_TEST_VALUE(Real, 0.158095863901207324355426285544321998253687969756843115) };

    BOOST_CHECK_CLOSE_FRACTION(Q.real(), Q_expected.real(), tol);
    BOOST_CHECK_CLOSE_FRACTION(Q.imag(), Q_expected.imag(), tol);
  }
}


BOOST_AUTO_TEST_CASE(sinh_sinh_quadrature_test)
{
    //
    // Uncomment the following to print out the coefficients:
    //
    /*
    std::cout << std::scientific << std::setprecision(8);
    print_levels(generate_constants<cpp_bin_float_100, float>(8), "f");
    std::cout << std::setprecision(18);
    print_levels(generate_constants<cpp_bin_float_100, double>(8), "");
    std::cout << std::setprecision(35);
    print_levels(generate_constants<cpp_bin_float_100, cpp_bin_float_quad>(8), "L");
    */
    test_nr_examples<float>();
    test_nr_examples<double>();
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    test_nr_examples<long double>();
#endif
    test_nr_examples<cpp_bin_float_quad>();
#if !defined(BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS) && !defined(BOOST_MATH_NO_REAL_CONCEPT_TESTS)
    test_nr_examples<boost::math::concepts::real_concept>();
#endif
#if !BOOST_WORKAROUND(BOOST_MSVC, < 1900) && defined(BOOST_MATH_RUN_MP_TESTS)
    test_nr_examples<boost::multiprecision::cpp_dec_float_50>();
#endif

    test_crc<float>();
    test_crc<double>();
    test_dirichlet_eta<std::complex<double>>();
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    test_crc<long double>();
    test_dirichlet_eta<std::complex<long double>>();
#endif
#ifdef BOOST_MATH_RUN_MP_TESTS
    test_crc<cpp_bin_float_quad>();
    test_dirichlet_eta<boost::multiprecision::cpp_complex_quad>();
#endif
#if !defined(BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS) && !defined(BOOST_MATH_NO_REAL_CONCEPT_TESTS)
    test_crc<boost::math::concepts::real_concept>();
#endif
#if !BOOST_WORKAROUND(BOOST_MSVC, < 1900) && defined(BOOST_MATH_RUN_MP_TESTS)
    test_crc<boost::multiprecision::cpp_dec_float_50>();
#endif


}
