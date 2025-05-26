// test file for quaternion.hpp

//  (C) Copyright Hubert Holin 2001.
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)


#include <iomanip>


#include <boost/mpl/list.hpp>

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/unit_test_log.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>

#include <boost/math/quaternion.hpp>

#ifdef _MSC_VER
#pragma warning(disable:4127) // conditional expression is constant
#endif

template<typename T>
struct string_type_name;

#define DEFINE_TYPE_NAME(Type)              \
template<> struct string_type_name<Type>    \
{                                           \
    static char const * _()                 \
    {                                       \
        return #Type;                       \
    }                                       \
}

DEFINE_TYPE_NAME(float);
DEFINE_TYPE_NAME(double);
DEFINE_TYPE_NAME(long double);
DEFINE_TYPE_NAME(boost::multiprecision::cpp_bin_float_quad);
DEFINE_TYPE_NAME(boost::multiprecision::number<boost::multiprecision::cpp_dec_float<25> >);

#if BOOST_WORKAROUND(BOOST_MSVC, < 1900)
#  define CPP_DEC_FLOAT_TEST
#else
#  define CPP_DEC_FLOAT_TEST , boost::multiprecision::number<boost::multiprecision::cpp_dec_float<25> >
#endif
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
#  define LD_TEST , long double
#else
#  define LD_TEST
#endif


typedef boost::mpl::list<float,double LD_TEST, boost::multiprecision::cpp_bin_float_quad CPP_DEC_FLOAT_TEST >  test_types;

// Apple GCC 4.0 uses the "double double" format for its long double,
// which means that epsilon is VERY small but useless for
// comparisons. So, don't do those comparisons.
#if (defined(__APPLE_CC__) && defined(__GNUC__) && __GNUC__ == 4) || defined(BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS)
typedef boost::mpl::list<float,double>  near_eps_test_types;
#else
typedef boost::mpl::list<float,double,long double>  near_eps_test_types;
#endif


#if BOOST_WORKAROUND(__GNUC__, < 3)
    // gcc 2.x ignores function scope using declarations,
    // put them in the scope of the enclosing namespace instead:
using   ::std::sqrt;
using   ::std::atan;
using   ::std::log;
using   ::std::exp;
using   ::std::cos;
using   ::std::sin;
using   ::std::tan;
using   ::std::cosh;
using   ::std::sinh;
using   ::std::tanh;

using   ::std::numeric_limits;

using   ::boost::math::abs;
#endif  /* BOOST_WORKAROUND(__GNUC__, < 3) */

#ifdef  BOOST_NO_STDC_NAMESPACE
using   ::sqrt;
using   ::atan;
using   ::log;
using   ::exp;
using   ::cos;
using   ::sin;
using   ::tan;
using   ::cosh;
using   ::sinh;
using   ::tanh;
#endif  /* BOOST_NO_STDC_NAMESPACE */

#ifdef  BOOST_NO_ARGUMENT_DEPENDENT_LOOKUP
using   ::boost::math::real;
using   ::boost::math::unreal;
using   ::boost::math::sup;
using   ::boost::math::l1;
using   ::boost::math::abs;
using   ::boost::math::norm;
using   ::boost::math::conj;
using   ::boost::math::exp;
using   ::boost::math::pow;
using   ::boost::math::cos;
using   ::boost::math::sin;
using   ::boost::math::tan;
using   ::boost::math::cosh;
using   ::boost::math::sinh;
using   ::boost::math::tanh;
using   ::boost::math::sinc_pi;
using   ::boost::math::sinhc_pi;
#endif  /* BOOST_NO_ARGUMENT_DEPENDENT_LOOKUP */
  
// Provide standard floating point abs() overloads if older Microsoft
// library is used with _MSC_EXTENSIONS defined. This code also works
// for the Intel compiler using the Microsoft library.
#if defined(_MSC_EXTENSIONS) && BOOST_WORKAROUND(_MSC_VER, < 1310)
#if !((__INTEL__ && _WIN32) && BOOST_WORKAROUND(__MWERKS__, >= 0x3201))
inline float        abs(float v)
{
    return(fabs(v));
}

inline double        abs(double v)
{
    return(fabs(v));
}

inline long double    abs(long double v)
{
    return(fabs(v));
}
#endif /* !((__INTEL__ && _WIN32) && BOOST_WORKAROUND(__MWERKS__, >= 0x3201)) */
#endif /* defined(_MSC_EXTENSIONS) && BOOST_WORKAROUND(_MSC_VER, < 1310) */


// explicit (if ludicrous) instanciation
#if !BOOST_WORKAROUND(__GNUC__, < 3)
template    class ::boost::math::quaternion<int>;
#else
// gcc doesn't like the absolutely-qualified namespace
template class boost::math::quaternion<int>;
#endif /* !BOOST_WORKAROUND(__GNUC__) */

template <class T>
struct other_type
{
   typedef double type;
};
template<>
struct other_type<double>
{
   typedef float type;
};
template<>
struct other_type<float>
{
   typedef short type;
};


template <class T, class U>
void test_compare(const T& a, const U& b, bool eq)
{
   if (eq)
   {
      BOOST_CHECK_EQUAL(a, b);
      BOOST_CHECK((a != b) == false);
      BOOST_CHECK_EQUAL(b, a);
      BOOST_CHECK((b != a) == false);
   }
   else
   {
      BOOST_CHECK_NE(a, b);
      BOOST_CHECK((a == b) == false);
      BOOST_CHECK_NE(b, a);
      BOOST_CHECK((b == a) == false);
   }
}

template <class T, class R1, class R2, class R3, class R4>
void check_exact_quaternion_result(const boost::math::quaternion<T>& q, R1 a, R2 b, R3 c, R4 d)
{
   BOOST_CHECK_EQUAL(q.R_component_1(), a);
   BOOST_CHECK_EQUAL(q.R_component_2(), b);
   BOOST_CHECK_EQUAL(q.R_component_3(), c);
   BOOST_CHECK_EQUAL(q.R_component_4(), d);
   BOOST_CHECK_EQUAL(q.C_component_1(), std::complex<T>(T(a), T(b)));
   BOOST_CHECK_EQUAL(q.C_component_2(), std::complex<T>(T(c), T(d)));
}

template <class T, class R1, class R2, class R3, class R4>
void check_approx_quaternion_result(const boost::math::quaternion<T>& q, R1 a, R2 b, R3 c, R4 d, int eps = 10)
{
   T tol = std::numeric_limits<T>::epsilon() * eps * 100;  // epsilon as a percentage.
   using std::abs;
   if (abs(a) > tol / 100)
   {
      BOOST_CHECK_CLOSE(q.R_component_1(), static_cast<T>(a), tol);
   }
   else
   {
      BOOST_CHECK_SMALL(q.R_component_1(), tol);
   }
   if (abs(b) > tol)
   {
      BOOST_CHECK_CLOSE(q.R_component_2(), static_cast<T>(b), tol);
   }
   else
   {
      BOOST_CHECK_SMALL(q.R_component_2(), tol);
   }
   if (abs(c) > tol)
   {
      BOOST_CHECK_CLOSE(q.R_component_3(), static_cast<T>(c), tol);
   }
   else
   {
      BOOST_CHECK_SMALL(q.R_component_3(), tol);
   }
   if (abs(d) > tol)
   {
      BOOST_CHECK_CLOSE(q.R_component_4(), static_cast<T>(d), tol);
   }
   else
   {
      BOOST_CHECK_SMALL(q.R_component_4(), tol);
   }
}

template <class T>
void check_complex_ops_imp()
{
   T tol = std::numeric_limits<T>::epsilon() * 200;

   ::std::complex<T>                   c0(5, 6);

   // using constructor "H seen as C^2"
   ::boost::math::quaternion<T>        q2(c0), q1;
   check_exact_quaternion_result(q2, 5, 6, 0, 0);

   // using converting assignment operator
   q2 = 0;
   q2 = c0;
   check_exact_quaternion_result(q2, 5, 6, 0, 0);

   // using += (const ::std::complex<T> &)
   q2 = ::boost::math::quaternion<T>(5, 6, 7, 8);
   q2 += c0;
   check_exact_quaternion_result(q2, 10, 12, 7, 8);

   // using -= (const ::std::complex<T> &)
   q2 -= c0;
   check_exact_quaternion_result(q2, 5, 6, 7, 8);

   // using *= (const ::std::complex<T> &)
   q2 *= c0;
   check_exact_quaternion_result(q2, -11, 60, 83, -2);

   q2 /= c0;
   check_approx_quaternion_result(q2, 5, 6, 7, 8);

   q2 = ::boost::math::quaternion<T>(4, 5, 7, 8);
   // operator +
   q1 = c0 + q2;
   check_exact_quaternion_result(q1, 9, 11, 7, 8);
   q1 = q2 + c0;
   check_exact_quaternion_result(q1, 9, 11, 7, 8);

   // operator -
   q1 = c0 - q2;
   check_exact_quaternion_result(q1, 1, 1, -7, -8);
   q1 = q2 - c0;
   check_exact_quaternion_result(q1, -1, -1, 7, 8);

   // using * (const ::std::complex<T> &, const quaternion<T> &)
   q1 = c0 * q2;
   check_exact_quaternion_result(q1, -10, 49, -13, 82);

   // using * (const quaternion<T> &, const ::std::complex<T> &)
   q1 = q2 * c0;
   check_exact_quaternion_result(q1, -10, 49, 83, -2);

   // using / (const ::std::complex<T> &, const quaternion<T> &)
   q1 = c0 / q2;
   check_approx_quaternion_result(q1, T(25) / 77, -T(1) / 154, T(13) / 154, -T(41) / 77);
   q1 *= q2;
   BOOST_CHECK_CLOSE(q1.R_component_1(), T(5), tol);
   BOOST_CHECK_CLOSE(q1.R_component_2(), T(6), tol);
   BOOST_CHECK_SMALL(q1.R_component_3(), tol);
   BOOST_CHECK_SMALL(q1.R_component_4(), tol);

   // using / (const quaternion<T> &, const ::std::complex<T> &)
   q1 = q2 / c0;
   check_approx_quaternion_result(q1, T(50) / 61, T(1)/ 61, -T(13) / 61, T(82) / 61);
   q1 *= c0;
   check_approx_quaternion_result(q1, 4, 5, 7, 8);

   q1 = c0;
   test_compare(q1, c0, true);
   q1 += 1;
   test_compare(q1, c0, false);
}

template <class T>
void check_complex_ops() {}

template<>
void check_complex_ops<float>() { check_complex_ops_imp<float>(); }
template<>
void check_complex_ops<double>() { check_complex_ops_imp<double>(); }
template<>
void check_complex_ops<long double>() { check_complex_ops_imp<long double>(); }

BOOST_AUTO_TEST_CASE_TEMPLATE(arithmetic_test, T, test_types)
{
   typedef typename other_type<T>::type other_type;
   check_complex_ops<T>();

   T tol = std::numeric_limits<T>::epsilon() * 200;

   // using default constructor
   ::boost::math::quaternion<T>        q0, q2;
   check_exact_quaternion_result(q0, 0, 0, 0, 0);
   BOOST_CHECK_EQUAL(q0, 0);

   ::boost::math::quaternion<T>        qa[2];
   check_exact_quaternion_result(qa[0], 0, 0, 0, 0);
   check_exact_quaternion_result(qa[1], 0, 0, 0, 0);
   BOOST_CHECK_EQUAL(qa[0], 0);
   BOOST_CHECK_EQUAL(qa[1], 0.f);

   // using constructor "H seen as R^4"
   ::boost::math::quaternion<T>       q1(1, 2, 3, 4);
   check_exact_quaternion_result(q1, 1, 2, 3, 4);

   // using untemplated copy constructor
   ::boost::math::quaternion<T>        q3(q1);
   check_exact_quaternion_result(q3, 1, 2, 3, 4);

   // using templated copy constructor
   ::boost::math::quaternion<other_type>  qo(5, 6, 7, 8);
   boost::math::quaternion<T>  q4(qo);
   check_exact_quaternion_result(q4, 5, 6, 7, 8);

   // using untemplated assignment operator
   q3 = q0;
   check_exact_quaternion_result(q0, 0, 0, 0, 0);
   //BOOST_CHECK_EQUAL(q3, 0.f);
   BOOST_CHECK_EQUAL(q3, q0);
   q3 = q4;
   check_exact_quaternion_result(q3, 5, 6, 7, 8);
   qa[0] = q4;
   check_exact_quaternion_result(qa[0], 5, 6, 7, 8);

   // using templated assignment operator
   q4 = qo;
   check_exact_quaternion_result(q4, 5, 6, 7, 8);

   other_type                                   f0(7);
   T                                            f1(7);

   // using converting assignment operator
   q2 = f0;
   check_exact_quaternion_result(q2, 7, 0, 0, 0);
   q2 = 33.;
   check_exact_quaternion_result(q2, 33, 0, 0, 0);

   // using += (const T &)
   q4 += f0;
   check_exact_quaternion_result(q4, 12, 6, 7, 8);

   // using += (const quaternion<X> &)
   q4 += q3;
   check_exact_quaternion_result(q4, 17, 12, 14, 16);

   // using -= (const T &)
   q4 -= f0;
   check_exact_quaternion_result(q4, 10, 12, 14, 16);

   // using -= (const quaternion<X> &)
   q4 -= q3;
   check_exact_quaternion_result(q4, 5, 6, 7, 8);

    // using *= (const T &)
   q4 *= f0;
   check_exact_quaternion_result(q4, 35, 42, 49, 56);
   // using *= (const quaternion<X> &)
   q4 *= q3;
   check_exact_quaternion_result(q4, -868, 420, 490, 560);

   // using /= (const T &)
   q4 /= f0;
   if(std::numeric_limits<T>::radix == 2)
      check_exact_quaternion_result(q4, -T(868) / 7, T(420) / 7, T(490) / 7, T(560) / 7);
   else
      // cpp_dec_float division is still inextact / not rounded:
      check_approx_quaternion_result(q4, -T(868) / 7, T(420) / 7, T(490) / 7, T(560) / 7);

   q4 = q3;
   q4 /= boost::math::quaternion<T>(9, 4, 6, 2);
   check_approx_quaternion_result(q4, T(127) / 137, T(68) / 137, T(13) / 137, T(54) / 137);
   q4 *= boost::math::quaternion<T>(9, 4, 6, 2);
   check_approx_quaternion_result(q4, 5, 6, 7, 8);

   q4 = boost::math::quaternion<T>(34, 56, 20, 2);
   // using + (const T &, const quaternion<T> &)
   q1 = f1 + q4;
   check_exact_quaternion_result(q1, 41, 56, 20, 2);

   // using + (const quaternion<T> &, const T &)
   q1 = q4 + f1;
   check_exact_quaternion_result(q1, 41, 56, 20, 2);

   // using + (const T &, const quaternion<T> &)
   q1 = f0 + q4;
   check_exact_quaternion_result(q1, 41, 56, 20, 2);

   // using + (const quaternion<T> &, const T &)
   q1 = q4 + f0;
   check_exact_quaternion_result(q1, 41, 56, 20, 2);

   // using + (const quaternion<T> &,const quaternion<T> &)
   q1 = q3 + q4;
   check_exact_quaternion_result(q1, 39, 62, 27, 10);

   // using - (const T &, const quaternion<T> &)
   q1 = f1 - q4;
   check_exact_quaternion_result(q1, 7-34, -56, -20, -2);

   // using - (const quaternion<T> &, const T &)
   q1 = q4 - f1;
   check_exact_quaternion_result(q1, 34-7, 56, 20, 2);

   // using - (const T &, const quaternion<T> &)
   q1 = f0 - q4;
   check_exact_quaternion_result(q1, 7-34, -56, -20, -2);

   // using - (const quaternion<T> &, const T &)
   q1 = q4 - f0;
   check_exact_quaternion_result(q1, 34-7, 56, 20, 2);

   // using - (const quaternion<T> &,const quaternion<T> &)
   q1 = q3 - q4;
   check_exact_quaternion_result(q1, -29, -50, -13, 6);

   // using * (const T &, const quaternion<T> &)
   q1 = f0 * q4;
   check_exact_quaternion_result(q1, 238, 392, 140, 14);

   // using * (const quaternion<T> &, const T &)
   q1 = q4 * f0;
   check_exact_quaternion_result(q1, 238, 392, 140, 14);

   // using * (const quaternion<T> &,const quaternion<T> &)
    q1 = q4 * q3;
   check_exact_quaternion_result(q1, -322, 630, -98, 554);
    q1 = q3 * q4;
   check_exact_quaternion_result(q1, -322, 338, 774, 10);

   // using / (const T &, const quaternion<T> &)
   q1 = T(f0) / q4;
   check_approx_quaternion_result(q1, T(119) / 2348, -T(49) / 587, -T(35) / 1174, -T(7) / 2348);
   q1 *= q4;
   BOOST_CHECK_CLOSE(q1.R_component_1(), T(7), tol);
   BOOST_CHECK_SMALL(q1.R_component_2(), tol);
   BOOST_CHECK_SMALL(q1.R_component_3(), tol);
   BOOST_CHECK_SMALL(q1.R_component_3(), tol);

   // using / (const quaternion<T> &, const T &)
   q1 = q4 / T(f0);
   check_approx_quaternion_result(q1, T(34) / 7, T(56) / 7, T(20) / 7, T(2) / 7);
    
   // using / (const quaternion<T> &,const quaternion<T> &)
   q1 = q4 / q3;
   check_approx_quaternion_result(q1, T(331) / 87, -T(35) / 87, T(149) / 87, -T(89) / 29);
   q1 *= q3;
   check_approx_quaternion_result(q1, 34, 56, 20, 2);

   // using + (const quaternion<T> &)
   q1 = +q4;
   check_exact_quaternion_result(q1, 34, 56, 20, 2);

   // using - (const quaternion<T> &)
   q1 = -q4;
   check_exact_quaternion_result(q1, -34, -56, -20, -2);

   // comparisons:
   q1 = f0;
   test_compare(q1, f0, true);
   q1 += 1;
   test_compare(q1, f0, false);
   q1 = q3;
   test_compare(q1, q3, true);
   q1 += 1;
   test_compare(q1, q3, false);

   #ifndef BOOST_MATH_STANDALONE
   BOOST_CHECK_EQUAL(boost::lexical_cast<std::string>(q4), "(34,56,20,2)");
   q1 = boost::lexical_cast<boost::math::quaternion<T> >("(34,56,20,2)");
   check_exact_quaternion_result(q1, 34, 56, 20, 2);

   q1 = q4 + 1;
   q1.swap(q4);
   check_exact_quaternion_result(q1, 34, 56, 20, 2);
   check_exact_quaternion_result(q4, 35, 56, 20, 2);
   swap(q1, q4);
   check_exact_quaternion_result(q1, 35, 56, 20, 2);
   check_exact_quaternion_result(q4, 34, 56, 20, 2);

   BOOST_CHECK_EQUAL(real(q4), 34);
   check_exact_quaternion_result(unreal(q1), 0, 56, 20, 2);
   BOOST_CHECK_EQUAL(sup(q4), 56);
   BOOST_CHECK_EQUAL(sup(-q4), 56);
   BOOST_CHECK_EQUAL(l1(q4), 34 + 56 + 20 + 2);
   BOOST_CHECK_EQUAL(l1(-q4), 34 + 56 + 20 + 2);
   BOOST_CHECK_CLOSE(abs(q4), boost::lexical_cast<T>("68.52736679604725626189080285032080446623"), tol);
   BOOST_CHECK_EQUAL(norm(q4), 4696);
   check_exact_quaternion_result(conj(q4), 34, -56, -20, -2);
   check_approx_quaternion_result(exp(q4), boost::lexical_cast<T>("-572700109350177.2871954597833265926769952"), boost::lexical_cast<T>("104986825963321.656891930274999993423955"), boost::lexical_cast<T>("37495294986900.59174711795535714050855537"), boost::lexical_cast<T>("3749529498690.059174711795535714050855537"), 300);
   check_approx_quaternion_result(pow(q4, 3), -321776, -4032, -1440, -144);
   check_approx_quaternion_result(sin(q4), boost::lexical_cast<T>("18285331065398556228976865.03309127394107"), boost::lexical_cast<T>("-27602822237164214909853379.68314411086089"), boost::lexical_cast<T>("-9858150798987219610661921.315408611021748"), boost::lexical_cast<T>("-985815079898721961066192.1315408611021748"), 40);
   check_approx_quaternion_result(cos(q4), boost::lexical_cast<T>("-29326963088663226843378365.81173441507358"), boost::lexical_cast<T>("-17210331032912252411431342.73890926302336"), boost::lexical_cast<T>("-6146546797468661575511193.835324736794056"), boost::lexical_cast<T>("-614654679746866157551119.3835324736794056"), 40);
   if(std::numeric_limits<T>::max_exponent >= std::numeric_limits<double>::max_exponent)
      check_approx_quaternion_result(tan(q4), boost::lexical_cast<T>("-3.758831069989140832054627039712718213887e-52"), boost::lexical_cast<T>("0.941209703633940052004990419391011076385"), boost::lexical_cast<T>("0.3361463227264071614303537212110753844232"), boost::lexical_cast<T>("0.03361463227264071614303537212110753844232"), 40);

   check_approx_quaternion_result(sinh(q4), boost::lexical_cast<T>("-286350054675088.6435977298916624551903343"), boost::lexical_cast<T>("52493412981660.82844596513750015091043914"), boost::lexical_cast<T>("18747647493450.29587355897767862532515683"), boost::lexical_cast<T>("1874764749345.029587355897767862532515683"), 200);
   check_approx_quaternion_result(cosh(q4), boost::lexical_cast<T>("-286350054675088.6435977298916641374866609"), boost::lexical_cast<T>("52493412981660.82844596513749984251351591"), boost::lexical_cast<T>("18747647493450.29587355897767851518339854"), boost::lexical_cast<T>("1874764749345.029587355897767851518339854"), 200);
   if(std::numeric_limits<T>::max_exponent >= std::numeric_limits<double>::max_exponent)
      check_approx_quaternion_result(tanh(q4), boost::lexical_cast<T>("0.9999999999999999999999999999945544805016"), boost::lexical_cast<T>("-2.075260044344318549117301019071435084233e-30"), boost::lexical_cast<T>("-7.411643015515423389704646496683696729404e-31"), boost::lexical_cast<T>("-7.411643015515423389704646496683696729404e-32"), 200);

#ifndef    BOOST_NO_TEMPLATE_TEMPLATES
   check_approx_quaternion_result(sinc_pi(q4), boost::lexical_cast<T>("-239180458943182912968898.352151239530846"), boost::lexical_cast<T>("-417903427539587405399855.0577257263862799"), boost::lexical_cast<T>("-149251224121281216214233.9491877594236714"), boost::lexical_cast<T>("-14925122412128121621423.39491877594236714"), 200);
   check_approx_quaternion_result(sinhc_pi(q4), boost::lexical_cast<T>("-1366603120232.604666248483234115586439226"), boost::lexical_cast<T>("3794799638667.255581055299959135992677524"), boost::lexical_cast<T>("1355285585238.305564662607128262854527687"), boost::lexical_cast<T>("135528558523.8305564662607128262854527687"), 200);
#endif
   #endif // BOOST_MATH_STANDALONE
   //
   // Construction variations:
   //
   T rho = boost::math::constants::root_two<T>() * 2;
   T theta = boost::math::constants::pi<T>() / 4;
   T phi1 = theta;
   T phi2 = theta;
   q1 = ::boost::math::spherical(rho, theta, phi1, phi2);
   check_approx_quaternion_result(q1, 1, 1, boost::math::constants::root_two<T>(), 2, 10);
   T alpha = theta;
   q1 = ::boost::math::semipolar(rho, alpha, phi1, phi2);
   check_approx_quaternion_result(q1, boost::math::constants::root_two<T>(), boost::math::constants::root_two<T>(), boost::math::constants::root_two<T>(), boost::math::constants::root_two<T>(), 10);
   T rho1 = 1;
   T rho2 = 2;
   T theta1 = 0;
   T theta2 = boost::math::constants::half_pi<T>();
   q1 = ::boost::math::multipolar(rho1, theta1, rho2, theta2);
   check_approx_quaternion_result(q1, 1, 0, 0, 2, 10);
   T t = 5;
   T radius = boost::math::constants::root_two<T>();
   T longitude = boost::math::constants::pi<T>() / 4;
   T latitude = boost::math::constants::pi<T>() / 3;
   q1 = ::boost::math::cylindrospherical(t, radius, longitude, latitude);

   #ifndef BOOST_MATH_STANDALONE
   check_approx_quaternion_result(q1, 5, 0.5, 0.5, boost::lexical_cast<T>("1.224744871391589049098642037352945695983"), 10);
   #endif
   
   T r = boost::math::constants::root_two<T>();
   T angle = boost::math::constants::pi<T>() / 4;
   T h1 = 3;
   T h2 = 4;
   q1 = ::boost::math::cylindrical(r, angle, h1, h2);
   check_approx_quaternion_result(q1, 1, 1, 3, 4, 10);

   ::boost::math::quaternion<T>        quaternion_1(1);
   ::boost::math::quaternion<T>        quaternion_i(0, 1);
   ::boost::math::quaternion<T>        quaternion_j(0, 0, 1);
   ::boost::math::quaternion<T>        quaternion_k(0, 0, 0, 1);
   check_exact_quaternion_result(quaternion_1 * quaternion_1, 1, 0, 0, 0);
   check_exact_quaternion_result(quaternion_1 * quaternion_i, 0, 1, 0, 0);
   check_exact_quaternion_result(quaternion_1 * quaternion_j, 0, 0, 1, 0);
   check_exact_quaternion_result(quaternion_1 * quaternion_k, 0, 0, 0, 1);

   check_exact_quaternion_result(quaternion_i * quaternion_1, 0, 1, 0, 0);
   check_exact_quaternion_result(quaternion_i * quaternion_i, -1, 0, 0, 0);
   check_exact_quaternion_result(quaternion_i * quaternion_j, 0, 0, 0, 1);
   check_exact_quaternion_result(quaternion_i * quaternion_k, 0, 0, -1, 0);

   check_exact_quaternion_result(quaternion_j * quaternion_1, 0, 0, 1, 0);
   check_exact_quaternion_result(quaternion_j * quaternion_i, 0, 0, 0, -1);
   check_exact_quaternion_result(quaternion_j * quaternion_j, -1, 0, 0, 0);
   check_exact_quaternion_result(quaternion_j * quaternion_k, 0, 1, 0, 0);

   check_exact_quaternion_result(quaternion_k * quaternion_1, 0, 0, 0, 1);
   check_exact_quaternion_result(quaternion_k * quaternion_i, 0, 0, 1, 0);
   check_exact_quaternion_result(quaternion_k * quaternion_j, 0, -1, 0, 0);
   check_exact_quaternion_result(quaternion_k * quaternion_k, -1, 0, 0, 0);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(multiplication_test, T, test_types)
{
#if     BOOST_WORKAROUND(__GNUC__, < 3)
#else   /* BOOST_WORKAROUND(__GNUC__, < 3) */
    using ::std::numeric_limits;
    
    using ::boost::math::abs;
#endif  /* BOOST_WORKAROUND(__GNUC__, < 3) */
    
    
    BOOST_TEST_MESSAGE("Testing multiplication for "
        << string_type_name<T>::_() << ".");
    
    BOOST_REQUIRE_PREDICATE(::std::less_equal<T>(),
        (abs(::boost::math::quaternion<T>(1,0,0,0)*
             ::boost::math::quaternion<T>(1,0,0,0)-static_cast<T>(1)))
        (numeric_limits<T>::epsilon()));
    
    BOOST_REQUIRE_PREDICATE(::std::less_equal<T>(),
        (abs(::boost::math::quaternion<T>(0,1,0,0)*
             ::boost::math::quaternion<T>(0,1,0,0)+static_cast<T>(1)))
        (numeric_limits<T>::epsilon()));
    
    BOOST_REQUIRE_PREDICATE(::std::less_equal<T>(),
        (abs(::boost::math::quaternion<T>(0,0,1,0)*
             ::boost::math::quaternion<T>(0,0,1,0)+static_cast<T>(1)))
        (numeric_limits<T>::epsilon()));
    
    BOOST_REQUIRE_PREDICATE(::std::less_equal<T>(),
        (abs(::boost::math::quaternion<T>(0,0,0,1)*
             ::boost::math::quaternion<T>(0,0,0,1)+static_cast<T>(1)))
        (numeric_limits<T>::epsilon()));
    
    BOOST_REQUIRE_PREDICATE(::std::less_equal<T>(),
        (abs(::boost::math::quaternion<T>(0,1,0,0)*
             ::boost::math::quaternion<T>(0,0,1,0)-
             ::boost::math::quaternion<T>(0,0,0,1)))
        (numeric_limits<T>::epsilon()));
    
    BOOST_REQUIRE_PREDICATE(::std::less_equal<T>(),
        (abs(::boost::math::quaternion<T>(0,0,1,0)*
             ::boost::math::quaternion<T>(0,1,0,0)+
             ::boost::math::quaternion<T>(0,0,0,1)))
        (numeric_limits<T>::epsilon()));
    
    BOOST_REQUIRE_PREDICATE(::std::less_equal<T>(),
        (abs(::boost::math::quaternion<T>(0,0,1,0)*
             ::boost::math::quaternion<T>(0,0,0,1)-
             ::boost::math::quaternion<T>(0,1,0,0)))
        (numeric_limits<T>::epsilon()));
    
    BOOST_REQUIRE_PREDICATE(::std::less_equal<T>(),
        (abs(::boost::math::quaternion<T>(0,0,0,1)*
             ::boost::math::quaternion<T>(0,0,1,0)+
             ::boost::math::quaternion<T>(0,1,0,0)))
        (numeric_limits<T>::epsilon()));
    
    BOOST_REQUIRE_PREDICATE(::std::less_equal<T>(),
        (abs(::boost::math::quaternion<T>(0,0,0,1)*
             ::boost::math::quaternion<T>(0,1,0,0)-
             ::boost::math::quaternion<T>(0,0,1,0)))
        (numeric_limits<T>::epsilon()));
    
    BOOST_REQUIRE_PREDICATE(::std::less_equal<T>(),
        (abs(::boost::math::quaternion<T>(0,1,0,0)*
             ::boost::math::quaternion<T>(0,0,0,1)+
             ::boost::math::quaternion<T>(0,0,1,0)))
        (numeric_limits<T>::epsilon()));
}


BOOST_AUTO_TEST_CASE_TEMPLATE(exp_test, T, test_types)
{
#if     BOOST_WORKAROUND(__GNUC__, < 3)
#else   /* BOOST_WORKAROUND(__GNUC__, < 3) */
    using ::std::numeric_limits;
    
    using ::std::atan;
    
    using ::boost::math::abs;
#endif  /* BOOST_WORKAROUND(__GNUC__, < 3) */
    
    
    BOOST_TEST_MESSAGE("Testing exp for "
        << string_type_name<T>::_() << ".");
    
    BOOST_CHECK_PREDICATE(::std::less_equal<T>(),
        (abs(exp(::boost::math::quaternion<T>
             (0,4*atan(static_cast<T>(1)),0,0))+static_cast<T>(1)))
        (2*numeric_limits<T>::epsilon()));
    
    BOOST_CHECK_PREDICATE(::std::less_equal<T>(),
        (abs(exp(::boost::math::quaternion<T>
             (0,0,4*atan(static_cast<T>(1)),0))+static_cast<T>(1)))
        (2*numeric_limits<T>::epsilon()));
    
    BOOST_CHECK_PREDICATE(::std::less_equal<T>(),
        (abs(exp(::boost::math::quaternion<T>
             (0,0,0,4*atan(static_cast<T>(1))))+static_cast<T>(1)))
        (2*numeric_limits<T>::epsilon()));
}


BOOST_AUTO_TEST_CASE_TEMPLATE(cos_test, T, test_types)
{
#if     BOOST_WORKAROUND(__GNUC__, < 3)
#else   /* BOOST_WORKAROUND(__GNUC__, < 3) */
    using ::std::numeric_limits;
    
    using ::std::log;
    
    using ::boost::math::abs;
#endif  /* BOOST_WORKAROUND(__GNUC__, < 3) */
    
    
    BOOST_TEST_MESSAGE("Testing cos for "
        << string_type_name<T>::_() << ".");
    
    BOOST_CHECK_PREDICATE(::std::less_equal<T>(),
        (abs(static_cast<T>(4)*cos(::boost::math::quaternion<T>
             (0,log(static_cast<T>(2)),0,0))-static_cast<T>(5)))
        (4*numeric_limits<T>::epsilon()));
    
    BOOST_CHECK_PREDICATE(::std::less_equal<T>(),
        (abs(static_cast<T>(4)*cos(::boost::math::quaternion<T>
             (0,0,log(static_cast<T>(2)),0))-static_cast<T>(5)))
        (4*numeric_limits<T>::epsilon()));
    
    BOOST_CHECK_PREDICATE(::std::less_equal<T>(),
        (abs(static_cast<T>(4)*cos(::boost::math::quaternion<T>
             (0,0,0,log(static_cast<T>(2))))-static_cast<T>(5)))
        (4*numeric_limits<T>::epsilon()));
}


BOOST_AUTO_TEST_CASE_TEMPLATE(sin_test, T, test_types)
{
#if     BOOST_WORKAROUND(__GNUC__, < 3)
#else   /* BOOST_WORKAROUND(__GNUC__, < 3) */
    using ::std::numeric_limits;
    
    using ::std::log;
    
    using ::boost::math::abs;
#endif  /* BOOST_WORKAROUND(__GNUC__, < 3) */
    
    
    BOOST_TEST_MESSAGE("Testing sin for "
        << string_type_name<T>::_() << ".");
    
    BOOST_CHECK_PREDICATE(::std::less_equal<T>(),
        (abs(static_cast<T>(4)*sin(::boost::math::quaternion<T>
             (0,log(static_cast<T>(2)),0,0))
             -::boost::math::quaternion<T>(0,3,0,0)))
        (4*numeric_limits<T>::epsilon()));
    
    BOOST_CHECK_PREDICATE(::std::less_equal<T>(),
        (abs(static_cast<T>(4)*sin(::boost::math::quaternion<T>
             (0,0,log(static_cast<T>(2)),0))
             -::boost::math::quaternion<T>(0,0,3,0)))
        (4*numeric_limits<T>::epsilon()));
    
    BOOST_CHECK_PREDICATE(::std::less_equal<T>(),
        (abs(static_cast<T>(4)*sin(::boost::math::quaternion<T>
             (0,0,0,log(static_cast<T>(2))))
             -::boost::math::quaternion<T>(0,0,0,3)))
        (4*numeric_limits<T>::epsilon()));
}


BOOST_AUTO_TEST_CASE_TEMPLATE(cosh_test, T, test_types)
{
#if     BOOST_WORKAROUND(__GNUC__, < 3)
#else   /* BOOST_WORKAROUND(__GNUC__, < 3) */
    using ::std::numeric_limits;
    
    using ::std::atan;
    
    using ::boost::math::abs;
#endif  /* BOOST_WORKAROUND(__GNUC__, < 3) */
    
    
    BOOST_TEST_MESSAGE("Testing cosh for "
        << string_type_name<T>::_() << ".");
    
    BOOST_CHECK_PREDICATE(::std::less_equal<T>(),
        (abs(cosh(::boost::math::quaternion<T>
             (0,4*atan(static_cast<T>(1)),0,0))
             +static_cast<T>(1)))
        (4*numeric_limits<T>::epsilon()));
    
    BOOST_CHECK_PREDICATE(::std::less_equal<T>(),
        (abs(cosh(::boost::math::quaternion<T>
             (0,0,4*atan(static_cast<T>(1)),0))
             +static_cast<T>(1)))
        (4*numeric_limits<T>::epsilon()));
    
    BOOST_CHECK_PREDICATE(::std::less_equal<T>(),
        (abs(cosh(::boost::math::quaternion<T>
             (0,0,0,4*atan(static_cast<T>(1))))
             +static_cast<T>(1)))
        (4*numeric_limits<T>::epsilon()));
}


BOOST_AUTO_TEST_CASE_TEMPLATE(sinh_test, T, test_types)
{
#if     BOOST_WORKAROUND(__GNUC__, < 3)
#else   /* BOOST_WORKAROUND(__GNUC__, < 3) */
    using ::std::numeric_limits;
    
    using ::std::atan;
    
    using ::boost::math::abs;
#endif  /* BOOST_WORKAROUND(__GNUC__, < 3) */
    
    
    BOOST_TEST_MESSAGE("Testing sinh for "
        << string_type_name<T>::_() << ".");
    
    BOOST_CHECK_PREDICATE(::std::less_equal<T>(),
        (abs(sinh(::boost::math::quaternion<T>
             (0,2*atan(static_cast<T>(1)),0,0))
             -::boost::math::quaternion<T>(0,1,0,0)))
        (4*numeric_limits<T>::epsilon()));
    
    BOOST_CHECK_PREDICATE(::std::less_equal<T>(),
        (abs(sinh(::boost::math::quaternion<T>
             (0,0,2*atan(static_cast<T>(1)),0))
             -::boost::math::quaternion<T>(0,0,1,0)))
        (4*numeric_limits<T>::epsilon()));
    
    BOOST_CHECK_PREDICATE(::std::less_equal<T>(),
        (abs(sinh(::boost::math::quaternion<T>
             (0,0,0,2*atan(static_cast<T>(1))))
             -::boost::math::quaternion<T>(0,0,0,1)))
        (4*numeric_limits<T>::epsilon()));
}

