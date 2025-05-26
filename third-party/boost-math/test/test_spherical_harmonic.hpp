//  (C) Copyright John Maddock 2007.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_MATH_OVERFLOW_ERROR_POLICY ignore_error
#include <boost/math/concepts/real_concept.hpp>
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/array.hpp>
#include <type_traits>
#include "functor.hpp"

#include "handle_test_result.hpp"
#include "table_type.hpp"


template <class Real, class T>
void do_test_spherical_harmonic(const T& data, const char* type_name, const char* test_name)
{
   typedef Real                   value_type;

   typedef value_type(*pg)(unsigned, int, value_type, value_type);
#ifdef SPHERICAL_HARMONIC_R_FUNCTION_TO_TEST
   pg funcp = SPHERICAL_HARMONIC_R_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   pg funcp = boost::math::spherical_harmonic_r<value_type, value_type>;
#else
   pg funcp = boost::math::spherical_harmonic_r;
#endif

   boost::math::tools::test_result<value_type> result;

   std::cout << "Testing " << test_name << " with type " << type_name
      << "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

   //
   // test Spheric Harmonic against data:
   //
#if !(defined(ERROR_REPORTING_MODE) && !defined(SPHERICAL_HARMONIC_R_FUNCTION_TO_TEST))
   result = boost::math::tools::test_hetero<Real>(
      data,
      bind_func_int2<Real>(funcp, 0, 1, 2, 3),
      extract_result<Real>(4));
   handle_test_result(result, data[result.worst()], result.worst(), type_name, "spherical_harmonic_r", test_name);
#endif

#if !(defined(ERROR_REPORTING_MODE) && !defined(SPHERICAL_HARMONIC_I_FUNCTION_TO_TEST))

#ifdef SPHERICAL_HARMONIC_I_FUNCTION_TO_TEST
   funcp = SPHERICAL_HARMONIC_I_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   funcp = boost::math::spherical_harmonic_i<value_type, value_type>;
#else
   funcp = boost::math::spherical_harmonic_i;
#endif
   //
   // test Spheric Harmonic against data:
   //
   result = boost::math::tools::test_hetero<Real>(
      data,
      bind_func_int2<Real>(funcp, 0, 1, 2, 3),
      extract_result<Real>(5));
   handle_test_result(result, data[result.worst()], result.worst(), type_name, "spherical_harmonic_i", test_name);

   std::cout << std::endl;
#endif
}

template <class Real, class T>
void test_complex_spherical_harmonic(const T& data, const char* /* name */, std::true_type const &)
{
   typedef Real                   value_type;

   for(unsigned i = 0; i < sizeof(data) / sizeof(data[0]); ++i)
   {
      //
      // Sanity check that the complex version does the same thing as the real
      // and imaginary versions:
      //
      std::complex<value_type> r = boost::math::spherical_harmonic(
         boost::math::tools::real_cast<unsigned>(data[i][0]),
         boost::math::tools::real_cast<unsigned>(data[i][1]),
         Real(data[i][2]),
         Real(data[i][3]));
      value_type re = boost::math::spherical_harmonic_r(
         boost::math::tools::real_cast<unsigned>(data[i][0]),
         boost::math::tools::real_cast<unsigned>(data[i][1]),
         Real(data[i][2]),
         Real(data[i][3]));
      value_type im = boost::math::spherical_harmonic_i(
         boost::math::tools::real_cast<unsigned>(data[i][0]),
         boost::math::tools::real_cast<unsigned>(data[i][1]),
         Real(data[i][2]),
         Real(data[i][3]));
      BOOST_CHECK_CLOSE_FRACTION(std::real(r), re, value_type(5));
      BOOST_CHECK_CLOSE_FRACTION(std::imag(r), im, value_type(5));
   }
}

template <class Real, class T>
void test_complex_spherical_harmonic(const T& /* data */, const char* /* name */, std::false_type const &)
{
   // T is not a built in type, can't use std::complex with it...
}

template <class T>
void test_spherical_harmonic(T, const char* name)
{
   //
   // The actual test data is rather verbose, so it's in a separate file
   //
   // The contents are as follows, each row of data contains
   // 6 items, the 4 input values, plus the real and imaginary results:
   //
#  include "spherical_harmonic.ipp"

   do_test_spherical_harmonic<T>(spherical_harmonic, name, "Spherical Harmonics");

   test_complex_spherical_harmonic<T>(spherical_harmonic, name, typename std::is_floating_point<T>::type());
}

template <class T>
void test_spots(T, const char* t)
{
   std::cout << "Testing basic sanity checks for type " << t << std::endl;
   //
   // basic sanity checks, tolerance is 100 epsilon:
   //
   T tolerance = boost::math::tools::epsilon<T>() * 100;
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_r(3, 2, static_cast<T>(0.5), static_cast<T>(0)), static_cast<T>(0.2061460599687871330692286791802688341213L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_r(20, 10, static_cast<T>(0.75), static_cast<T>(-0.25)), static_cast<T>(0.06197787102219208244041677775577045124092L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_i(20, 10, static_cast<T>(0.75), static_cast<T>(-0.25)), static_cast<T>(0.04629885158895932341185988759669916977920L), tolerance);

   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_r(40, 15, static_cast<T>(-0.75), static_cast<T>(2.25)), static_cast<T>(0.2806904825045745687343492963236868973484L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_i(40, 15, static_cast<T>(-0.75), static_cast<T>(2.25)), static_cast<T>(-0.2933918444656603582282372590387544902135L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_r(40, 15, static_cast<T>(-0.75), static_cast<T>(-2.25)), static_cast<T>(0.2806904825045745687343492963236868973484L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_i(40, 15, static_cast<T>(-0.75), static_cast<T>(-2.25)), static_cast<T>(0.2933918444656603582282372590387544902135L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_r(40, 15, static_cast<T>(0.75), static_cast<T>(-2.25)), static_cast<T>(-0.2806904825045745687343492963236868973484L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_i(40, 15, static_cast<T>(0.75), static_cast<T>(-2.25)), static_cast<T>(-0.2933918444656603582282372590387544902135L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_r(40, 15, static_cast<T>(0.75), static_cast<T>(2.25)), static_cast<T>(-0.2806904825045745687343492963236868973484L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_i(40, 15, static_cast<T>(0.75), static_cast<T>(2.25)), static_cast<T>(0.2933918444656603582282372590387544902135L), tolerance);

   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_r(20, 14, static_cast<T>(-0.75), static_cast<T>(2.25)), static_cast<T>(0.3479218186133435466692822481919867452442L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_i(20, 14, static_cast<T>(-0.75), static_cast<T>(2.25)), static_cast<T>(0.0293201066685263879566422194539567289974L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_r(20, 14, static_cast<T>(-0.75), static_cast<T>(-2.25)), static_cast<T>(0.3479218186133435466692822481919867452442L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_i(20, 14, static_cast<T>(-0.75), static_cast<T>(-2.25)), static_cast<T>(-0.0293201066685263879566422194539567289974L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_r(20, 14, static_cast<T>(0.75), static_cast<T>(-2.25)), static_cast<T>(0.3479218186133435466692822481919867452442L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_i(20, 14, static_cast<T>(0.75), static_cast<T>(-2.25)), static_cast<T>(-0.0293201066685263879566422194539567289974L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_r(20, 14, static_cast<T>(0.75), static_cast<T>(2.25)), static_cast<T>(0.3479218186133435466692822481919867452442L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_i(20, 14, static_cast<T>(0.75), static_cast<T>(2.25)), static_cast<T>(0.0293201066685263879566422194539567289974L), tolerance);

   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_r(39, 15, static_cast<T>(-0.75), static_cast<T>(2.25)), static_cast<T>(0.1757594233240278196989039119899901986211L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_i(39, 15, static_cast<T>(-0.75), static_cast<T>(2.25)), static_cast<T>(-0.1837126108841860058078729532035715580790L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_r(39, 15, static_cast<T>(-0.75), static_cast<T>(-2.25)), static_cast<T>(0.1757594233240278196989039119899901986211L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_i(39, 15, static_cast<T>(-0.75), static_cast<T>(-2.25)), static_cast<T>(0.1837126108841860058078729532035715580790L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_r(39, 15, static_cast<T>(0.75), static_cast<T>(-2.25)), static_cast<T>(-0.1757594233240278196989039119899901986211L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_i(39, 15, static_cast<T>(0.75), static_cast<T>(-2.25)), static_cast<T>(-0.1837126108841860058078729532035715580790L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_r(39, 15, static_cast<T>(0.75), static_cast<T>(2.25)), static_cast<T>(-0.1757594233240278196989039119899901986211L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_i(39, 15, static_cast<T>(0.75), static_cast<T>(2.25)), static_cast<T>(0.1837126108841860058078729532035715580790L), tolerance);

   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_r(19, 14, static_cast<T>(-0.75), static_cast<T>(2.25)), static_cast<T>(0.2341701030303444033808969389588343934828L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_i(19, 14, static_cast<T>(-0.75), static_cast<T>(2.25)), static_cast<T>(0.0197340092863212879172432610952871202640L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_r(19, 14, static_cast<T>(-0.75), static_cast<T>(-2.25)), static_cast<T>(0.2341701030303444033808969389588343934828L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_i(19, 14, static_cast<T>(-0.75), static_cast<T>(-2.25)), static_cast<T>(-0.0197340092863212879172432610952871202640L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_r(19, 14, static_cast<T>(0.75), static_cast<T>(-2.25)), static_cast<T>(0.2341701030303444033808969389588343934828L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_i(19, 14, static_cast<T>(0.75), static_cast<T>(-2.25)), static_cast<T>(-0.0197340092863212879172432610952871202640L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_r(19, 14, static_cast<T>(0.75), static_cast<T>(2.25)), static_cast<T>(0.2341701030303444033808969389588343934828L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_i(19, 14, static_cast<T>(0.75), static_cast<T>(2.25)), static_cast<T>(0.0197340092863212879172432610952871202640L), tolerance);

   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_r(40, -15, static_cast<T>(-0.75), static_cast<T>(2.25)), static_cast<T>(-0.2806904825045745687343492963236868973484L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_i(40, -15, static_cast<T>(-0.75), static_cast<T>(2.25)), static_cast<T>(-0.2933918444656603582282372590387544902135L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_r(40, -15, static_cast<T>(-0.75), static_cast<T>(-2.25)), static_cast<T>(-0.2806904825045745687343492963236868973484L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_i(40, -15, static_cast<T>(-0.75), static_cast<T>(-2.25)), static_cast<T>(0.2933918444656603582282372590387544902135L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_r(40, -15, static_cast<T>(0.75), static_cast<T>(-2.25)), static_cast<T>(0.2806904825045745687343492963236868973484L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_i(40, -15, static_cast<T>(0.75), static_cast<T>(-2.25)), static_cast<T>(-0.2933918444656603582282372590387544902135L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_r(40, -15, static_cast<T>(0.75), static_cast<T>(2.25)), static_cast<T>(0.2806904825045745687343492963236868973484L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_i(40, -15, static_cast<T>(0.75), static_cast<T>(2.25)), static_cast<T>(0.2933918444656603582282372590387544902135L), tolerance);

   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_r(20, -14, static_cast<T>(-0.75), static_cast<T>(2.25)), static_cast<T>(0.3479218186133435466692822481919867452442L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_i(20, -14, static_cast<T>(-0.75), static_cast<T>(2.25)), static_cast<T>(-0.0293201066685263879566422194539567289974L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_r(20, -14, static_cast<T>(-0.75), static_cast<T>(-2.25)), static_cast<T>(0.3479218186133435466692822481919867452442L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_i(20, -14, static_cast<T>(-0.75), static_cast<T>(-2.25)), static_cast<T>(0.0293201066685263879566422194539567289974L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_r(20, -14, static_cast<T>(0.75), static_cast<T>(-2.25)), static_cast<T>(0.3479218186133435466692822481919867452442L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_i(20, -14, static_cast<T>(0.75), static_cast<T>(-2.25)), static_cast<T>(0.0293201066685263879566422194539567289974L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_r(20, -14, static_cast<T>(0.75), static_cast<T>(2.25)), static_cast<T>(0.3479218186133435466692822481919867452442L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_i(20, -14, static_cast<T>(0.75), static_cast<T>(2.25)), static_cast<T>(-0.0293201066685263879566422194539567289974L), tolerance);

   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_r(20, 14, static_cast<T>(-4), static_cast<T>(2.25)), static_cast<T>(0.5253373768014719124617844890495875474590L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_i(20, 14, static_cast<T>(-4), static_cast<T>(2.25)), static_cast<T>(0.0442712905622650144694916590407495495699L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_r(20, 14, static_cast<T>(-4), static_cast<T>(-2.25)), static_cast<T>(0.5253373768014719124617844890495875474590L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_i(20, 14, static_cast<T>(-4), static_cast<T>(-2.25)), static_cast<T>(-0.0442712905622650144694916590407495495699L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_r(20, 14, static_cast<T>(4), static_cast<T>(-2.25)), static_cast<T>(0.5253373768014719124617844890495875474590L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_i(20, 14, static_cast<T>(4), static_cast<T>(-2.25)), static_cast<T>(-0.0442712905622650144694916590407495495699L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_r(20, 14, static_cast<T>(4), static_cast<T>(2.25)), static_cast<T>(0.5253373768014719124617844890495875474590L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_i(20, 14, static_cast<T>(4), static_cast<T>(2.25)), static_cast<T>(0.0442712905622650144694916590407495495699L), tolerance);

   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_r(20, 15, static_cast<T>(-4), static_cast<T>(2.25)), static_cast<T>(-0.2991140325667575801827063718821420263438L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_i(20, 15, static_cast<T>(-4), static_cast<T>(2.25)), static_cast<T>(0.3126490678888350710506307405826667514065L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_r(20, 15, static_cast<T>(-4), static_cast<T>(-2.25)), static_cast<T>(-0.2991140325667575801827063718821420263438L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_i(20, 15, static_cast<T>(-4), static_cast<T>(-2.25)), static_cast<T>(-0.3126490678888350710506307405826667514065L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_r(20, 15, static_cast<T>(4), static_cast<T>(-2.25)), static_cast<T>(0.2991140325667575801827063718821420263438L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_i(20, 15, static_cast<T>(4), static_cast<T>(-2.25)), static_cast<T>(0.3126490678888350710506307405826667514065L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_r(20, 15, static_cast<T>(4), static_cast<T>(2.25)), static_cast<T>(0.2991140325667575801827063718821420263438L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_i(20, 15, static_cast<T>(4), static_cast<T>(2.25)), static_cast<T>(-0.3126490678888350710506307405826667514065L), tolerance);

   BOOST_CHECK_EQUAL(::boost::math::spherical_harmonic_r(10, 15, static_cast<T>(-0.75), static_cast<T>(2.25)), static_cast<T>(0));
   BOOST_CHECK_EQUAL(::boost::math::spherical_harmonic_i(10, 15, static_cast<T>(-0.75), static_cast<T>(2.25)), static_cast<T>(0));
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_r(53, 42, static_cast<T>(-8.75), static_cast<T>(-2.25)), static_cast<T>(-0.0008147976618889536159592309471859037113647L), tolerance);
   BOOST_CHECK_CLOSE_FRACTION(::boost::math::spherical_harmonic_i(53, 42, static_cast<T>(-8.75), static_cast<T>(-2.25)), static_cast<T>(0.0002099802242493057018193798824353982612756L), tolerance);
}
