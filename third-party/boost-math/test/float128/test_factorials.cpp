//  Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifdef _MSC_VER
#  pragma warning(disable: 4127) // conditional expression is constant.
#  pragma warning(disable: 4245) // int/unsigned int conversion
#endif

// Return infinities not exceptions:
#define BOOST_MATH_OVERFLOW_ERROR_POLICY ignore_error

#include <boost/cstdfloat.hpp>
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/special_functions/factorials.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/tools/stats.hpp>
#include <boost/math/tools/test.hpp>

#include <iostream>
  using std::cout;
  using std::endl;

#if LDBL_MANT_DIG != 113

template <class T>
T naive_falling_factorial(T x, unsigned n)
{
   if(n == 0)
      return 1;
   T result = x;
   while(--n)
   {
      x -= 1;
      result *= x;
   }
   return result;
}

template <class T>
void test_spots(T)
{
   //
   // Basic sanity checks.
   //
   T tolerance = boost::math::tools::epsilon<T>() * 100 * 2;  // 2 eps as a percent.
   BOOST_CHECK_CLOSE(
      ::boost::math::factorial<T>(0),
      static_cast<T>(1), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::factorial<T>(1),
      static_cast<T>(1), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::factorial<T>(10),
      static_cast<T>(3628800L), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::unchecked_factorial<T>(0),
      static_cast<T>(1), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::unchecked_factorial<T>(1),
      static_cast<T>(1), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::unchecked_factorial<T>(10),
      static_cast<T>(3628800L), tolerance);

   //
   // Try some double factorials:
   //
   BOOST_CHECK_CLOSE(
      ::boost::math::double_factorial<T>(0),
      static_cast<T>(1), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::double_factorial<T>(1),
      static_cast<T>(1), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::double_factorial<T>(2),
      static_cast<T>(2), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::double_factorial<T>(5),
      static_cast<T>(15), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::double_factorial<T>(10),
      static_cast<T>(3840), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::double_factorial<T>(19),
      static_cast<T>(6.547290750e8Q), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::double_factorial<T>(24),
      static_cast<T>(1.961990553600000e12Q), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::double_factorial<T>(33),
      static_cast<T>(6.33265987076285062500000e18Q), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::double_factorial<T>(42),
      static_cast<T>(1.0714547155728479551488000000e26Q), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::double_factorial<T>(47),
      static_cast<T>(1.19256819277443412353990764062500000e30Q), tolerance);

   if((std::numeric_limits<T>::has_infinity) && (std::numeric_limits<T>::max_exponent <= 1024))
   {
      BOOST_CHECK_EQUAL(
         ::boost::math::double_factorial<T>(320),
         std::numeric_limits<T>::infinity());
      BOOST_CHECK_EQUAL(
         ::boost::math::double_factorial<T>(301),
         std::numeric_limits<T>::infinity());
   }
   //
   // Rising factorials:
   //
   tolerance = boost::math::tools::epsilon<T>() * 100 * 20;  // 20 eps as a percent.
   if(std::numeric_limits<T>::is_specialized == 0)
      tolerance *= 5;  // higher error rates without Lanczos support
   BOOST_CHECK_CLOSE(
      ::boost::math::rising_factorial(static_cast<T>(3), 4),
      static_cast<T>(360), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::rising_factorial(static_cast<T>(7), -4),
      static_cast<T>(0.00277777777777777777777777777777777777777777777777777777777778Q), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::rising_factorial(static_cast<T>(120.5f), 8),
      static_cast<T>(5.58187566784927180664062500e16Q), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::rising_factorial(static_cast<T>(120.5f), -4),
      static_cast<T>(5.15881498170104646868208445266116850161120996179812063177241e-9Q), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::rising_factorial(static_cast<T>(5000.25f), 8),
      static_cast<T>(3.92974581976666067544013393509103775024414062500000e29Q), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::rising_factorial(static_cast<T>(5000.25f), -7),
      static_cast<T>(1.28674092710208810281923019294164707555099052561945725535047e-26Q), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::rising_factorial(static_cast<T>(30.25), 21),
      static_cast<T>(3.93286957998925490693364184100209193343633629069699964020401e33Q), tolerance * 2);
   BOOST_CHECK_CLOSE(
      ::boost::math::rising_factorial(static_cast<T>(30.25), -21),
      static_cast<T>(3.35010902064291983728782493133164809108646650368560147505884e-27Q), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::rising_factorial(static_cast<T>(-30.25), 21),
      static_cast<T>(-9.76168312768123676601980433377916854311706629232503473758698e26Q), tolerance * 2);
   BOOST_CHECK_CLOSE(
      ::boost::math::rising_factorial(static_cast<T>(-30.25), -21),
      static_cast<T>(-1.50079704000923674318934280259377728203516775215430875839823e-34Q), 2 * tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::rising_factorial(static_cast<T>(-30.25), 5),
      static_cast<T>(-1.78799177197265625000000e7Q), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::rising_factorial(static_cast<T>(-30.25), -5),
      static_cast<T>(-2.47177487004482195012362027432181137141899692171397467859150e-8Q), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::rising_factorial(static_cast<T>(-30.25), 6),
      static_cast<T>(4.5146792242309570312500000e8Q), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::rising_factorial(static_cast<T>(-30.25), -6),
      static_cast<T>(6.81868929667537089689274558433603136943171564610751635473516e-10Q), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::rising_factorial(static_cast<T>(-3), 6),
      static_cast<T>(0), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::rising_factorial(static_cast<T>(-3.25), 6),
      static_cast<T>(2.99926757812500Q), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::rising_factorial(static_cast<T>(-5.25), 6),
      static_cast<T>(50.987548828125000000000000Q), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::rising_factorial(static_cast<T>(-5.25), 13),
      static_cast<T>(127230.91046623885631561279296875000Q), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::rising_factorial(static_cast<T>(-3.25), -6),
      static_cast<T>(0.0000129609865918182348202632178291407500332449622510474437452125Q), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::rising_factorial(static_cast<T>(-5.25), -6),
      static_cast<T>(2.50789821857946332294524052303699065683926911849535903362649e-6Q), tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::rising_factorial(static_cast<T>(-5.25), -13),
      static_cast<T>(-1.38984989447269128946284683518361786049649013886981662962096e-14Q), tolerance);

   //
   // Falling factorials:
   //
   BOOST_CHECK_CLOSE(
      ::boost::math::falling_factorial(static_cast<T>(30.25), 0),
      static_cast<T>(naive_falling_factorial(30.25Q, 0)),
      tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::falling_factorial(static_cast<T>(30.25), 1),
      static_cast<T>(naive_falling_factorial(30.25Q, 1)),
      tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::falling_factorial(static_cast<T>(30.25), 2),
      static_cast<T>(naive_falling_factorial(30.25Q, 2)),
      tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::falling_factorial(static_cast<T>(30.25), 5),
      static_cast<T>(naive_falling_factorial(30.25Q, 5)),
      tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::falling_factorial(static_cast<T>(30.25), 22),
      static_cast<T>(naive_falling_factorial(30.25Q, 22)),
      tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::falling_factorial(static_cast<T>(100.5), 6),
      static_cast<T>(naive_falling_factorial(100.5Q, 6)),
      tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::falling_factorial(static_cast<T>(30.75), 30),
      static_cast<T>(naive_falling_factorial(30.75Q, 30)),
      tolerance * 3);
   if(boost::math::policies::digits<T, boost::math::policies::policy<> >() > 50)
   {
      BOOST_CHECK_CLOSE(
         ::boost::math::falling_factorial(static_cast<T>(-30.75Q), 30),
         static_cast<T>(naive_falling_factorial(-30.75Q, 30)),
         tolerance * 3);
      BOOST_CHECK_CLOSE(
         ::boost::math::falling_factorial(static_cast<T>(-30.75Q), 27),
         static_cast<T>(naive_falling_factorial(-30.75Q, 27)),
         tolerance * 3);
   }
   BOOST_CHECK_CLOSE(
      ::boost::math::falling_factorial(static_cast<T>(-12.0), 6),
      static_cast<T>(naive_falling_factorial(-12.0Q, 6)),
      tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::falling_factorial(static_cast<T>(-12), 5),
      static_cast<T>(naive_falling_factorial(-12.0Q, 5)),
      tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::falling_factorial(static_cast<T>(-3.0), 6),
      static_cast<T>(naive_falling_factorial(-3.0Q, 6)),
      tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::falling_factorial(static_cast<T>(-3), 5),
      static_cast<T>(naive_falling_factorial(-3.0Q, 5)),
      tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::falling_factorial(static_cast<T>(3.0), 6),
      static_cast<T>(naive_falling_factorial(3.0Q, 6)),
      tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::falling_factorial(static_cast<T>(3), 5),
      static_cast<T>(naive_falling_factorial(3.0Q, 5)),
      tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::falling_factorial(static_cast<T>(3.25), 4),
      static_cast<T>(naive_falling_factorial(3.25Q, 4)),
      tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::falling_factorial(static_cast<T>(3.25), 5),
      static_cast<T>(naive_falling_factorial(3.25Q, 5)),
      tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::falling_factorial(static_cast<T>(3.25), 6),
      static_cast<T>(naive_falling_factorial(3.25Q, 6)),
      tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::falling_factorial(static_cast<T>(3.25), 7),
      static_cast<T>(naive_falling_factorial(3.25Q, 7)),
      tolerance);
   BOOST_CHECK_CLOSE(
      ::boost::math::falling_factorial(static_cast<T>(8.25), 12),
      static_cast<T>(naive_falling_factorial(8.25Q, 12)),
      tolerance);


   tolerance = boost::math::tools::epsilon<T>() * 100 * 20;  // 20 eps as a percent.
   unsigned i = boost::math::max_factorial<T>::value;
   if((boost::is_floating_point<T>::value) && (sizeof(T) <= sizeof(double)))
   {
      // Without Lanczos support, tgamma isn't accurate enough for this test:
      BOOST_CHECK_CLOSE(
         ::boost::math::unchecked_factorial<T>(i),
         boost::math::tgamma(static_cast<T>(i+1)), tolerance);
   }

   i += 10;
   while(boost::math::lgamma(static_cast<T>(i+1)) < boost::math::tools::log_max_value<T>())
   {
      BOOST_CHECK_CLOSE(
         ::boost::math::factorial<T>(i),
         boost::math::tgamma(static_cast<T>(i+1)), tolerance);
      i += 10;
   }
} // template <class T> void test_spots(T)
#endif

BOOST_AUTO_TEST_CASE( test_main )
{
   BOOST_MATH_CONTROL_FP;
#if LDBL_MANT_DIG != 113
   test_spots(0.0Q);
   cout << "max factorial for __float128"  << boost::math::max_factorial<boost::floatmax_t>::value  << endl;
#endif
}



