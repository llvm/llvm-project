//  Copyright John Maddock 2012
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef SYCL_LANGUAGE_VERSION
#include <pch_light.hpp>
#endif

#define BOOST_MATH_OVERFLOW_ERROR_POLICY ignore_error
#include <boost/math/tools/config.hpp>

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/math/concepts/real_concept.hpp>
#include <boost/array.hpp>
#include <boost/math/special_functions/hankel.hpp>
#include <iostream>
#include <iomanip>

#ifdef _MSC_VER
#  pragma warning(disable : 4756 4127) // overflow in constant arithmetic
// Constants are too big for float case, but this doesn't matter for test.
#endif

//
// DESCRIPTION:
// ~~~~~~~~~~~~
//
// This file tests the Hankel H1 and H2 functions.  
// These are basically just a few spot tests, since the underlying implementation
// is the same as for the Bessel functions.
//
template <class T>
void check_close(const std::complex<T>& a, const std::complex<T>& b)
{
   T tol = boost::math::tools::epsilon<T>() * 3000;
   BOOST_CHECK_CLOSE_FRACTION(a.real(), b.real(), tol);
   BOOST_CHECK_CLOSE_FRACTION(a.imag(), b.imag(), tol);
}

template <class T>
void test_hankel(T, const char* name)
{
   std::cout << "Testing type " << name << std::endl;

   static const std::array<std::array<std::complex<T>, 4>, 16> data = 
   {{
      // Values are v, z, J, and Y.
      // H1 and H2 are calculated from functions.wolfram.com.
      {{ 0, 1, static_cast<T>(0.765197686557966551449717526102663220909274289755325241861548L), static_cast<T>(0.0882569642156769579829267660235151628278175230906755467110438L) }},
      {{ 20, 15.5, static_cast<T>(0.0114685669049540880132573889173945340441319848974792913849131L), static_cast<T>(-2.23357703561803241011539677726233479433825678625142168545338L) }},
      {{ 202, 65, static_cast<T>(4.03123907894335908039069177339845754619082182624766366629474e-77L), static_cast<T>(-4.12853948338543234456930697819285129000267780751856135651604e73L) }},
      {{ 1.25, 2.25, static_cast<T>(0.548918751190427983177518806565451279306141559548962787802891L), static_cast<T>(-0.125900744882628421758627761371862848797979705547465002154794L) }},
      {{ -20, 15.5, static_cast<T>(0.0114685669049540880132573889173945340441319848974792913849131L), static_cast<T>(-2.23357703561803241011539677726233479433825678625142168545338L) }},
      {{ -20, -15.5,  static_cast<T>(0.0114685669049540880132573889173945340441319848974792913849131L), std::complex<T>(static_cast<T>(-2.23357703561803241011539677726233479433825678625142168545338L), static_cast<T>(0.02293713380990817602651477783478906808826396979495858276983L))}},
      {{ 1.25, -1.5, std::complex<T>(static_cast<T>(-0.335713500965919366139805990226845134897000581426417882156072L),  static_cast<T>(-0.335713500965919366139805990226845134897000581426417882156072L)), std::complex<T>(static_cast<T>(0.392747687664481978664420363126684555843062241632017696636768L), static_cast<T>(-1.064174689596320710944032343580374825637063404484853460948911L)) }},
      {{ -1.25, -1.5, std::complex<T>(static_cast<T>(0.515099846311769525023685454389374962206960751452481078650112L),  static_cast<T>(-0.515099846311769525023685454389374962206960751452481078650112L)), std::complex<T>(static_cast<T>(-0.040329260174013212604062774398331485097569895404765001721544L), static_cast<T>(0.989870432449525837443308134380418439316351607500197155578680L)) }},
      {{ 4, 4, static_cast<T>(0.281129064961360106322277160229942806897088617059328870629222L), static_cast<T>(-0.488936768533842510615657398339913206218740182079627974737267L) }},
      {{ 4, -4, static_cast<T>(0.281129064961360106322277160229942806897088617059328870629222L), std::complex<T>(static_cast<T>(-0.488936768533842510615657398339913206218740182079627974737267L), static_cast<T>(0.562258129922720212644554320459885613794177234118657741258443L)) }},
      {{ -4, 4, static_cast<T>(0.281129064961360106322277160229942806897088617059328870629222L), static_cast<T>(-0.488936768533842510615657398339913206218740182079627974737267L) }},
      {{ -4, -4, static_cast<T>(0.281129064961360106322277160229942806897088617059328870629222L), std::complex<T>(static_cast<T>(-0.488936768533842510615657398339913206218740182079627974737267L), static_cast<T>(0.562258129922720212644554320459885613794177234118657741258443L)) }},
      {{ 3, 3, static_cast<T>(0.309062722255251643618260194946833149429135935993056794354475L), static_cast<T>(-0.538541616105031618004703905338594463807957863604859251481262L) }},
      {{ 3, -3, static_cast<T>(-0.309062722255251643618260194946833149429135935993056794354475L), std::complex<T>(static_cast<T>(0.538541616105031618004703905338594463807957863604859251481262L), static_cast<T>(-0.618125444510503287236520389893666298858271871986113588708949L)) }},
      {{ -3, 3, static_cast<T>(-0.309062722255251643618260194946833149429135935993056794354475L), static_cast<T>(0.538541616105031618004703905338594463807957863604859251481262L) }},
      {{ -3, -3, static_cast<T>(0.309062722255251643618260194946833149429135935993056794354475L), std::complex<T>(static_cast<T>(-0.538541616105031618004703905338594463807957863604859251481262L), static_cast<T>(0.618125444510503287236520389893666298858271871986113588708949L)) }},
   }};

   std::complex<T> im(0, 1);
   for(unsigned i = 0; i < data.size(); ++i)
   {
      if((i != 2) || (std::numeric_limits<T>::max_exponent10 > 80))
      {
         check_close(boost::math::cyl_hankel_1(data[i][0].real(), data[i][1].real()), data[i][2] + im * data[i][3]);
         check_close(boost::math::cyl_hankel_2(data[i][0].real(), data[i][1].real()), data[i][2] - im * data[i][3]);

         check_close(
            boost::math::cyl_hankel_1(data[i][0].real() + 0.5f, data[i][1].real()) * boost::math::constants::root_half_pi<T>() / sqrt(data[i][1]),
            boost::math::sph_hankel_1(data[i][0].real(), data[i][1].real()));
         check_close(
            boost::math::cyl_hankel_2(data[i][0].real() + 0.5f, data[i][1].real()) * boost::math::constants::root_half_pi<T>() / sqrt(data[i][1]),
            boost::math::sph_hankel_2(data[i][0].real(), data[i][1].real()));
      }
   }
}

//
// Instantiate a few instances to check our error handling code can cope with std::complex:
//
#ifndef SYCL_LANGUAGE_VERSION
typedef boost::math::policies::policy<
   boost::math::policies::overflow_error<boost::math::policies::throw_on_error>,
   boost::math::policies::denorm_error<boost::math::policies::throw_on_error>,
   boost::math::policies::underflow_error<boost::math::policies::throw_on_error>,
   boost::math::policies::domain_error<boost::math::policies::throw_on_error>,
   boost::math::policies::pole_error<boost::math::policies::throw_on_error>,
   boost::math::policies::rounding_error<boost::math::policies::throw_on_error>,
   boost::math::policies::evaluation_error<boost::math::policies::throw_on_error>,
   boost::math::policies::indeterminate_result_error<boost::math::policies::throw_on_error> > pol1;

template std::complex<double> boost::math::cyl_hankel_1<double, double, pol1>(double, double, const pol1&);

typedef boost::math::policies::policy<
   boost::math::policies::overflow_error<boost::math::policies::errno_on_error>,
   boost::math::policies::denorm_error<boost::math::policies::errno_on_error>,
   boost::math::policies::underflow_error<boost::math::policies::errno_on_error>,
   boost::math::policies::domain_error<boost::math::policies::errno_on_error>,
   boost::math::policies::pole_error<boost::math::policies::errno_on_error>,
   boost::math::policies::rounding_error<boost::math::policies::errno_on_error>,
   boost::math::policies::evaluation_error<boost::math::policies::errno_on_error>,
   boost::math::policies::indeterminate_result_error<boost::math::policies::errno_on_error> > pol2;

template std::complex<double> boost::math::cyl_hankel_1<double, double, pol2>(double, double, const pol2&);

typedef boost::math::policies::policy<
   boost::math::policies::overflow_error<boost::math::policies::ignore_error>,
   boost::math::policies::denorm_error<boost::math::policies::ignore_error>,
   boost::math::policies::underflow_error<boost::math::policies::ignore_error>,
   boost::math::policies::domain_error<boost::math::policies::ignore_error>,
   boost::math::policies::pole_error<boost::math::policies::ignore_error>,
   boost::math::policies::rounding_error<boost::math::policies::ignore_error>,
   boost::math::policies::evaluation_error<boost::math::policies::ignore_error>,
   boost::math::policies::indeterminate_result_error<boost::math::policies::ignore_error> > pol3;

template std::complex<double> boost::math::cyl_hankel_1<double, double, pol3>(double, double, const pol3&);
#endif

BOOST_AUTO_TEST_CASE( test_main )
{
#ifdef TEST_GSL
   gsl_set_error_handler_off();
#endif
   BOOST_MATH_CONTROL_FP;

#ifndef BOOST_MATH_BUGGY_LARGE_FLOAT_CONSTANTS
   test_hankel(0.1F, "float");
#endif
   test_hankel(0.1, "double");
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   test_hankel(0.1L, "long double");
#else
   std::cout << "<note>The long double tests have been disabled on this platform "
      "either because the long double overloads of the usual math functions are "
      "not available at all, or because they are too inaccurate for these tests "
      "to pass.</note>" << std::endl;
#endif
   
}




