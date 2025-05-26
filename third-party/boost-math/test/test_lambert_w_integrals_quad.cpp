// Copyright Paul A. Bristow 2016, 2017, 2018.
// Copyright John Maddock 2016.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// test_lambert_w_integrals.cpp
//! \brief quadrature tests that cover the whole range of the Lambert W0 function.

#include <boost/config.hpp>   // for BOOST_MSVC definition etc.
#include <boost/version.hpp>   // for BOOST_MSVC versions.

// Boost macros
#define BOOST_TEST_MAIN
#define BOOST_LIB_DIAGNOSTIC "on" // Report library file details.
#include <boost/test/included/unit_test.hpp> // Boost.Test
// #include <boost/test/unit_test.hpp> // Boost.Test
#include <boost/test/tools/floating_point_comparison.hpp>

#include <boost/array.hpp>
#include <boost/type_traits/is_constructible.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
using boost::multiprecision::cpp_bin_float_quad;

#include <boost/math/special_functions/fpclassify.hpp> // isnan, isfinite.
#include <boost/math/special_functions/next.hpp> // float_next, float_prior
using boost::math::float_next;
using boost::math::float_prior;
#include <boost/math/special_functions/ulp.hpp>  // ulp

#include <boost/math/tools/test_value.hpp>  // for create_test_value and macro BOOST_MATH_TEST_VALUE.
#include <boost/math/policies/policy.hpp>
using boost::math::policies::digits2;
using boost::math::policies::digits10;
#include <boost/math/special_functions/lambert_w.hpp> // For Lambert W lambert_w function.
using boost::math::lambert_wm1;
using boost::math::lambert_w0;

#include <limits>
#include <cmath>
#include <typeinfo>
#include <iostream>
#include <type_traits>
#include <exception>

std::string show_versions(void);

// Added code and test for Integral of the Lambert W function: by Nick Thompson.
// https://en.wikipedia.org/wiki/Lambert_W_function#Definite_integrals

#include <boost/math/constants/constants.hpp> // for integral tests.
#include <boost/math/quadrature/tanh_sinh.hpp> // for integral tests.
#include <boost/math/quadrature/exp_sinh.hpp> // for integral tests.

  using boost::math::policies::policy;
  using boost::math::policies::make_policy;

// using statements needed for changing error handling policy.
using boost::math::policies::evaluation_error;
using boost::math::policies::domain_error;
using boost::math::policies::overflow_error;
using boost::math::policies::ignore_error;
using boost::math::policies::throw_on_error;

typedef policy<
  domain_error<throw_on_error>,
  overflow_error<ignore_error>
> no_throw_policy;

// Assumes that function has a throw policy, for example:
//    NOT lambert_w0<T>(1 / (x * x), no_throw_policy());
// Error in function boost::math::quadrature::exp_sinh<double>::integrate:
// The exp_sinh quadrature evaluated your function at a singular point and resulted in inf.
// Please ensure your function evaluates to a finite number of its entire domain.
template <typename T>
T debug_integration_proc(T x)
{
   T result; // warning C4701: potentially uninitialized local variable 'result' used
  // T result = 0 ; // But result may not be assigned below?
  try
  {
   // Assign function call to result in here...
    if (x <= sqrt(boost::math::tools::min_value<T>()) )
    {
      result = 0;
    }
    else
    {
      result = lambert_w0<T>(1 / (x * x));
    }
   // result = lambert_w0<T>(1 / (x * x), no_throw_policy());  // Bad idea, less helpful diagnostic message is:
    // Error in function boost::math::quadrature::exp_sinh<double>::integrate:
    // The exp_sinh quadrature evaluated your function at a singular point and resulted in inf.
    // Please ensure your function evaluates to a finite number of its entire domain.

  } // try
  catch (const std::exception& e)
  {
    std::cout << "Exception " << e.what() << std::endl;
    // set breakpoint here:
    std::cout << "Unexpected exception thrown in integration code at abscissa (x): " << x << "." << std::endl;
    if (!std::isfinite(result))
    {
      // set breakpoint here:
      std::cout << "Unexpected non-finite result in integration code at abscissa (x): " << x << "." << std::endl;
    }
    if (std::isnan(result))
    {
      // set breakpoint here:
      std::cout << "Unexpected non-finite result in integration code at abscissa (x): " << x << "." << std::endl;
    }
  } // catch
  return result;
} // T debug_integration_proc(T x)

template<class Real>
void test_integrals()
{
  // Integral of the Lambert W function:
  // https://en.wikipedia.org/wiki/Lambert_W_function
  using boost::math::quadrature::tanh_sinh;
  using boost::math::quadrature::exp_sinh;
  // file:///I:/modular-boost/libs/math/doc/html/math_toolkit/quadrature/double_exponential/de_tanh_sinh.html
  using std::sqrt;

  std::cout << "Integration of type " << typeid(Real).name()  << std::endl;

  Real tol = std::numeric_limits<Real>::epsilon();
  { //  // Integrate for function lambert_W0(z);
    tanh_sinh<Real> ts;
    Real a = 0;
    Real b = boost::math::constants::e<Real>();
    auto f = [](Real z)->Real
    {
      return lambert_w0<Real>(z);
    };
    Real z = ts.integrate(f, a, b); // OK without any decltype(f)
    BOOST_CHECK_CLOSE_FRACTION(z, boost::math::constants::e<Real>() - 1, tol);
  }
  {
    // Integrate for function lambert_W0(z/(z sqrt(z)).
    exp_sinh<Real> es;
    auto f = [](Real z)->Real
    {
      return lambert_w0<Real>(z)/(z * sqrt(z));
    };
    Real z = es.integrate(f); // OK
    BOOST_CHECK_CLOSE_FRACTION(z, 2 * boost::math::constants::root_two_pi<Real>(), tol);
  }
  {
    // Integrate for function lambert_W0(1/z^2).
    exp_sinh<Real> es;
    //const Real sqrt_min = sqrt(boost::math::tools::min_value<Real>()); // 1.08420217e-19 fo 32-bit float.
    // error C3493: 'sqrt_min' cannot be implicitly captured because no default capture mode has been specified
    auto f = [](Real z)->Real
    {
      if (z <= sqrt(boost::math::tools::min_value<Real>()) )
      { // Too small would underflow z * z and divide by zero to overflow 1/z^2 for lambert_w0 z parameter.
        return static_cast<Real>(0);
      }
      else
      {
        return lambert_w0<Real>(1 / (z * z)); // warning C4756: overflow in constant arithmetic, even though cannot happen.
      }
    };
    Real z = es.integrate(f);
    BOOST_CHECK_CLOSE_FRACTION(z, boost::math::constants::root_two_pi<Real>(), tol);
  }
} // template<class Real> void test_integrals()


BOOST_AUTO_TEST_CASE( integrals )
{
  std::cout << "Macro BOOST_MATH_LAMBERT_W0_INTEGRALS is defined." << std::endl;
  BOOST_TEST_MESSAGE("\nTest Lambert W0 integrals.");
  try
  {
  // using statements needed to change precision policy.
  using boost::math::policies::policy;
  using boost::math::policies::make_policy;
  using boost::math::policies::precision;
  using boost::math::policies::digits2;
  using boost::math::policies::digits10;

  // using statements needed for changing error handling policy.
  using boost::math::policies::evaluation_error;
  using boost::math::policies::domain_error;
  using boost::math::policies::overflow_error;
  using boost::math::policies::ignore_error;
  using boost::math::policies::throw_on_error;

  /*
  typedef policy<
    domain_error<throw_on_error>,
    overflow_error<ignore_error>
  > no_throw_policy;

  // Experiment with better diagnostics.
  typedef float Real;

  Real inf = std::numeric_limits<Real>::infinity();
  Real max = (std::numeric_limits<Real>::max)();
  std::cout.precision(std::numeric_limits<Real>::max_digits10);
  //std::cout << "lambert_w0(inf) = " << lambert_w0(inf) << std::endl; // lambert_w0(inf) = 1.79769e+308
  std::cout << "lambert_w0(inf, throw_policy()) = " << lambert_w0(inf, no_throw_policy()) << std::endl; // inf
  std::cout << "lambert_w0(max) = " << lambert_w0(max) << std::endl; // lambert_w0(max) = 703.227
  //std::cout << lambert_w0(inf) << std::endl; // inf - will throw.
  std::cout << "lambert_w0(0) = " << lambert_w0(0.) << std::endl; // 0
  std::cout << "lambert_w0(std::numeric_limits<Real>::denorm_min()) = " << lambert_w0(std::numeric_limits<Real>::denorm_min()) << std::endl; // 4.94066e-324
  std::cout << "lambert_w0(std::numeric_limits<Real>::min()) = " << lambert_w0((std::numeric_limits<Real>::min)()) << std::endl; // 2.22507e-308

  // Approximate the largest lambert_w you can get for type T?
  float max_w_f = boost::math::lambert_w_detail::lambert_w0_approx((std::numeric_limits<float>::max)()); // Corless equation 4.19, page 349, and Chapeau-Blondeau equation 20, page 2162.
  std::cout << "w max_f " << max_w_f << std::endl; // 84.2879
  Real max_w = boost::math::lambert_w_detail::lambert_w0_approx((std::numeric_limits<Real>::max)()); // Corless equation 4.19, page 349, and Chapeau-Blondeau equation 20, page 2162.
  std::cout << "w max " << max_w << std::endl; // 703.227

  std::cout << "lambert_w0(7.2416706213544837e-163) = " << lambert_w0(7.2416706213544837e-163) << std::endl; //
  std::cout << "test integral 1/z^2" << std::endl;
  std::cout << "ULP = " << boost::math::ulp(1., policy<digits2<> >()) << std::endl; // ULP = 2.2204460492503131e-16
  std::cout << "ULP = " << boost::math::ulp(1e-10, policy<digits2<> >()) << std::endl; // ULP = 2.2204460492503131e-16
  std::cout << "ULP = " << boost::math::ulp(1., policy<digits2<11> >()) << std::endl; // ULP = 2.2204460492503131e-16
  std::cout << "epsilon =  " << std::numeric_limits<Real>::epsilon() << std::endl; //
  std::cout << "sqrt(max) =  " << sqrt(boost::math::tools::max_value<float>() ) << std::endl; // sqrt(max) =  1.8446742974197924e+19
  std::cout << "sqrt(min) =  " << sqrt(boost::math::tools::min_value<float>() ) << std::endl; // sqrt(min) =  1.0842021724855044e-19



// Demo debug version.
Real tol = std::numeric_limits<Real>::epsilon();
Real x;
{
  using boost::math::quadrature::exp_sinh;
  exp_sinh<Real> es;
  // Function to be integrated, lambert_w0(1/z^2).

    //auto f = [](Real z)->Real
    //{ // Naive - no protection against underflow and subsequent divide by zero.
    //  return lambert_w0<Real>(1 / (z * z));
    //};
    // Diagnostic is:
    // Error in function boost::math::lambert_w0<Real>: Expected a finite value but got inf

    auto f = [](Real z)->Real
    { // Debug with diagnostics for underflow and subsequent divide by zero and other bad things.
      return debug_integration_proc(z);
    };
    // Exception Error in function boost::math::lambert_w0<double>: Expected a finite value but got inf.

    // Unexpected exception thrown in integration code at abscissa: 7.2416706213544837e-163.
    // Unexpected exception thrown in integration code at abscissa (x): 3.478765835953569e-23.
    x = es.integrate(f);
    std::cout << "es.integrate(f) = " << x << std::endl;
    BOOST_CHECK_CLOSE_FRACTION(x, boost::math::constants::root_two_pi<Real>(), tol);
    // root_two_pi<double = 2.506628274631000502
  }
    */

  test_integrals<cpp_bin_float_quad>();
  }
  catch (std::exception& ex)
  {
    std::cout << ex.what() << std::endl;
  }
}

