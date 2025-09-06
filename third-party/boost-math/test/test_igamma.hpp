// Copyright John Maddock 2006.
// Copyright Paul A. Bristow 2007, 2009
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_MATH_OVERFLOW_ERROR_POLICY ignore_error

#include <boost/math/concepts/real_concept.hpp>
#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/math/special_functions/gamma.hpp>
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include "../include_private/boost/math/tools/test.hpp"
#include <boost/math/tools/stats.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/type_traits/is_floating_point.hpp>
#include <boost/array.hpp>
#include "functor.hpp"

#include "handle_test_result.hpp"
#include "table_type.hpp"

#ifndef SC_
#define SC_(x) static_cast<typename table_type<T>::type>(BOOST_JOIN(x, L))
#endif

template <class Real, class T>
void do_test_gamma_2(const T& data, const char* type_name, const char* test_name)
{
   typedef Real                   value_type;

   typedef value_type (*pg)(value_type, value_type);
   pg funcp;

   boost::math::tools::test_result<value_type> result;

#if !(defined(ERROR_REPORTING_MODE) && !defined(IGAMMA_FUNCTION_TO_TEST))

#ifdef IGAMMA_FUNCTION_TO_TEST
   funcp = IGAMMA_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   funcp = boost::math::tgamma<value_type, value_type>;
#else
   funcp = boost::math::tgamma;
#endif

   std::cout << "Testing " << test_name << " with type " << type_name
      << "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

   //
   // test tgamma(T, T) against data:
   //
   if(Real(data[0][2]) > 0)
   {
      result = boost::math::tools::test_hetero<Real>(
         data,
         bind_func<Real>(funcp, 0, 1),
         extract_result<Real>(2));
      handle_test_result(result, data[result.worst()], result.worst(), type_name, "tgamma (incomplete)", test_name);
      //
      // test tgamma_lower(T, T) against data:
      //
#ifdef IGAMMAL_FUNCTION_TO_TEST
      funcp = IGAMMAL_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
      funcp = boost::math::tgamma_lower<value_type, value_type>;
#else
      funcp = boost::math::tgamma_lower;
#endif
      result = boost::math::tools::test_hetero<Real>(
         data,
         bind_func<Real>(funcp, 0, 1),
         extract_result<Real>(4));
      handle_test_result(result, data[result.worst()], result.worst(), type_name, "tgamma_lower", test_name);
   }
#endif
#if !(defined(ERROR_REPORTING_MODE) && !defined(GAMMAQ_FUNCTION_TO_TEST))
   //
   // test gamma_q(T, T) against data:
   //
#ifdef GAMMAQ_FUNCTION_TO_TEST
   funcp = GAMMAQ_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   funcp = boost::math::gamma_q<value_type, value_type>;
#else
   funcp = boost::math::gamma_q;
#endif
   result = boost::math::tools::test_hetero<Real>(
      data,
      bind_func<Real>(funcp, 0, 1),
      extract_result<Real>(3));
   handle_test_result(result, data[result.worst()], result.worst(), type_name, "gamma_q", test_name);
   //
   // test gamma_p(T, T) against data:
   //
#ifdef GAMMAP_FUNCTION_TO_TEST
   funcp = GAMMAP_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   funcp = boost::math::gamma_p<value_type, value_type>;
#else
   funcp = boost::math::gamma_p;
#endif
   result = boost::math::tools::test_hetero<Real>(
      data,
      bind_func<Real>(funcp, 0, 1),
      extract_result<Real>(5));
   handle_test_result(result, data[result.worst()], result.worst(), type_name, "gamma_p", test_name);
   std::cout << std::endl;
#endif
}

template <class T>
void test_gamma(T, const char* name)
{
   //
   // The actual test data is rather verbose, so it's in a separate file
   //
   // First the data for the incomplete gamma function, each
   // row has the following 6 entries:
   // Parameter a, parameter z,
   // Expected tgamma(a, z), Expected gamma_q(a, z)
   // Expected tgamma_lower(a, z), Expected gamma_p(a, z)
   //
#  include "igamma_med_data.ipp"

   do_test_gamma_2<T>(igamma_med_data, name, "tgamma(a, z) medium values");

#  include "igamma_small_data.ipp"

   do_test_gamma_2<T>(igamma_small_data, name, "tgamma(a, z) small values");

#  include "igamma_big_data.ipp"

   do_test_gamma_2<T>(igamma_big_data, name, "tgamma(a, z) large values");

#  include "igamma_int_data.ipp"

   do_test_gamma_2<T>(igamma_int_data, name, "tgamma(a, z) integer and half integer values");
}

template <class T>
void test_spots(T)
{
   //
   // basic sanity checks, tolerance is 10 epsilon expressed as a percentage:
   //
   T tolerance = boost::math::tools::epsilon<T>() * 1000;
#if (defined(macintosh) || defined(__APPLE__) || defined(__APPLE_CC__))
   tolerance *= 10;
#endif
   // An extra fudge factor for real_concept which has a less accurate tgamma:
   T tolerance_tgamma_extra = std::numeric_limits<T>::is_specialized ? 1 : 10;

   BOOST_CHECK_CLOSE(::boost::math::tgamma(static_cast<T>(5), static_cast<T>(1)), static_cast<T>(23.912163676143750903709045060494956383977723517065L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::tgamma(static_cast<T>(5), static_cast<T>(5)), static_cast<T>(10.571838841565097874621959975919877646444998907920L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::tgamma(static_cast<T>(5), static_cast<T>(10)), static_cast<T>(0.70206451384706574414638719662835463671916532623256L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::tgamma(static_cast<T>(5), static_cast<T>(100)), static_cast<T>(3.8734332808745531496973774140085644548465762343719e-36L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::tgamma(static_cast<T>(0.5), static_cast<T>(0.5)), static_cast<T>(0.56241823159440712427949495730204306902676756479651L), tolerance * tolerance_tgamma_extra);
   BOOST_CHECK_CLOSE(::boost::math::tgamma(static_cast<T>(0.5), static_cast<T>(9)/10), static_cast<T>(0.31853210360412109873859360390443790076576777747449L), tolerance*10);
   BOOST_CHECK_CLOSE(::boost::math::tgamma(static_cast<T>(0.5), static_cast<T>(5)), static_cast<T>(0.0027746032604128093194908357272603294120210079791437L), tolerance * tolerance_tgamma_extra);
   BOOST_CHECK_CLOSE(::boost::math::tgamma(static_cast<T>(0.5), static_cast<T>(100)), static_cast<T>(3.7017478604082789202535664481339075721362102520338e-45L), tolerance * tolerance_tgamma_extra);

   BOOST_CHECK_CLOSE(::boost::math::tgamma_lower(static_cast<T>(5), static_cast<T>(1)), static_cast<T>(0.087836323856249096290954939505043616022276482935091L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::tgamma_lower(static_cast<T>(5), static_cast<T>(5)), static_cast<T>(13.428161158434902125378040024080122353555001092080L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::tgamma_lower(static_cast<T>(5), static_cast<T>(10)), static_cast<T>(23.297935486152934255853612803371645363280834673767L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::tgamma_lower(static_cast<T>(5), static_cast<T>(100)), static_cast<T>(23.999999999999999999999999999999999996126566719125L), tolerance);

   BOOST_CHECK_CLOSE(::boost::math::gamma_q(static_cast<T>(5), static_cast<T>(1)), static_cast<T>(0.99634015317265628765454354418728984933240514654437L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::gamma_q(static_cast<T>(5), static_cast<T>(5)), static_cast<T>(0.44049328506521241144258166566332823526854162116334L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::gamma_q(static_cast<T>(5), static_cast<T>(10)), static_cast<T>(0.029252688076961072672766133192848109863298555259690L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::gamma_q(static_cast<T>(5), static_cast<T>(100)), static_cast<T>(1.6139305336977304790405739225035685228527400976549e-37L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::gamma_q(static_cast<T>(1.5), static_cast<T>(2)), static_cast<T>(0.26146412994911062220282207597592120190281060919079L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::gamma_q(static_cast<T>(20.5), static_cast<T>(22)), static_cast<T>(0.34575332043467326814971590879658406632570278929072L), tolerance);

   BOOST_CHECK_CLOSE(::boost::math::gamma_p(static_cast<T>(5), static_cast<T>(1)), static_cast<T>(0.0036598468273437123454564558127101506675948534556288L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::gamma_p(static_cast<T>(5), static_cast<T>(5)), static_cast<T>(0.55950671493478758855741833433667176473145837883666L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::gamma_p(static_cast<T>(5), static_cast<T>(10)), static_cast<T>(0.97074731192303892732723386680715189013670144474031L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::gamma_p(static_cast<T>(5), static_cast<T>(100)), static_cast<T>(0.9999999999999999999999999999999999998386069466302L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::gamma_p(static_cast<T>(1.5), static_cast<T>(2)), static_cast<T>(0.73853587005088937779717792402407879809718939080921L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::gamma_p(static_cast<T>(20.5), static_cast<T>(22)), static_cast<T>(0.65424667956532673185028409120341593367429721070928L), tolerance);

   // naive check on derivative function:
   using namespace std;  // For ADL of std functions
   tolerance = boost::math::tools::epsilon<T>() * 5000; // 50 eps
   BOOST_CHECK_CLOSE(::boost::math::gamma_p_derivative(static_cast<T>(20.5), static_cast<T>(22)), 
      exp(static_cast<T>(-22)) * pow(static_cast<T>(22), static_cast<T>(19.5)) / boost::math::tgamma(static_cast<T>(20.5)), tolerance);

   // Bug reports from Rocco Romeo:
   BOOST_CHECK_CLOSE(::boost::math::tgamma(static_cast<T>(20), ldexp(T(1), -40)), static_cast<T>(1.21645100408832000000e17L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::tgamma_lower(static_cast<T>(20), ldexp(T(1), -40)), static_cast<T>(7.498484069471659696438206828760307317022658816757448882e-243L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::gamma_p(static_cast<T>(20), ldexp(T(1), -40)), static_cast<T>(6.164230243774976473534975936127139110276824507876192062e-260L), tolerance);

   BOOST_CHECK_CLOSE(::boost::math::tgamma(static_cast<T>(30), ldexp(T(1), -30)), static_cast<T>(8.841761993739701954543616000000e30L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::tgamma_lower(static_cast<T>(30), ldexp(T(1), -30)), static_cast<T>(3.943507283668378474979245322638092813837393749566146974e-273L), tolerance);
#ifdef __SUNPRO_CC
   BOOST_CHECK_CLOSE(::boost::math::gamma_p(static_cast<T>(30), ldexp(T(1), -30)), static_cast<T>(4.460092102072560946444018923090222645613009128135650652e-304L), tolerance * 8);
#else
   BOOST_CHECK_CLOSE(::boost::math::gamma_p(static_cast<T>(30), ldexp(T(1), -30)), static_cast<T>(4.460092102072560946444018923090222645613009128135650652e-304L), tolerance);
#endif
   BOOST_CHECK_CLOSE(::boost::math::gamma_p_derivative(static_cast<T>(2), ldexp(T(1), -575)), static_cast<T>(8.08634922390438981326119906687585206568664784377654648227177e-174L), tolerance);

   //typedef boost::math::policies::policy<boost::math::policies::overflow_error<boost::math::policies::throw_on_error> > throw_policy;

   if(std::numeric_limits<T>::max_exponent <= 1024 && std::numeric_limits<T>::has_infinity)
   {
      BOOST_CHECK_EQUAL(::boost::math::tgamma(static_cast<T>(176), static_cast<T>(100)), std::numeric_limits<T>::infinity());
      //BOOST_MATH_CHECK_THROW(::boost::math::tgamma(static_cast<T>(176), static_cast<T>(100), throw_policy()), std::overflow_error);
      BOOST_CHECK_EQUAL(::boost::math::tgamma(static_cast<T>(530), static_cast<T>(2000)), std::numeric_limits<T>::infinity());
      BOOST_CHECK_EQUAL(::boost::math::tgamma(static_cast<T>(740), static_cast<T>(2500)), std::numeric_limits<T>::infinity());
      BOOST_CHECK_EQUAL(::boost::math::tgamma(static_cast<T>(530.5), static_cast<T>(2000)), std::numeric_limits<T>::infinity());
      BOOST_CHECK_EQUAL(::boost::math::tgamma(static_cast<T>(740.5), static_cast<T>(2500)), std::numeric_limits<T>::infinity());
      BOOST_CHECK_EQUAL(::boost::math::tgamma_lower(static_cast<T>(10000.0f), static_cast<T>(10000.0f / 4)), std::numeric_limits<T>::infinity());
   }
   if(std::numeric_limits<T>::max_exponent >= 1024)
   {
      BOOST_CHECK_CLOSE(::boost::math::tgamma(static_cast<T>(170), static_cast<T>(165)), static_cast<T>(2.737338337642022829223832094019477918166996032112404370e304L), (std::numeric_limits<T>::digits > 100 ? 10 : 3) * tolerance);
      BOOST_CHECK_CLOSE(::boost::math::tgamma_lower(static_cast<T>(170), static_cast<T>(165)), static_cast<T>(1.531729671362682445715419794880088619901822603944331733e304L), 3 * tolerance);
      BOOST_CHECK_CLOSE(::boost::math::tgamma(static_cast<T>(170), static_cast<T>(170)), static_cast<T>(2.090991698081449410761040647015858316167077909285580375e304L), 10 * tolerance);
      BOOST_CHECK_CLOSE(::boost::math::tgamma_lower(static_cast<T>(170), static_cast<T>(170)), static_cast<T>(2.178076310923255864178211241883708221901740726771155728e304L), 10 * tolerance);
      BOOST_CHECK_CLOSE(::boost::math::tgamma(static_cast<T>(170), static_cast<T>(190)), static_cast<T>(2.8359275512790301602903689596273175148895758522893941392e303L), 10 * tolerance);
      BOOST_CHECK_CLOSE(::boost::math::tgamma_lower(static_cast<T>(170), static_cast<T>(190)), static_cast<T>(3.985475253876802258910214992936834786579861050827796689e304L), 10 * tolerance);
      BOOST_CHECK_CLOSE(::boost::math::tgamma(static_cast<T>(170), static_cast<T>(1000)), static_cast<T>(6.1067635957780723069200425769800190368662985052038980542e72L), 10 * tolerance);

      BOOST_CHECK_CLOSE(::boost::math::tgamma_lower(static_cast<T>(185), static_cast<T>(1)), static_cast<T>(0.001999286058955490074702037576083582139834300307968257924836L), tolerance);
      BOOST_CHECK_CLOSE(::boost::math::tgamma(static_cast<T>(185), static_cast<T>(1500)), static_cast<T>(1.037189524841404054867100938934493979112615962865368623e-67L), tolerance * 10);

      BOOST_CHECK_CLOSE(::boost::math::tgamma(static_cast<T>(36), ldexp(static_cast<T>(1), -26)), static_cast<T>(1.03331479663861449296666513375232000000e40L), tolerance * 10);
      BOOST_CHECK_CLOSE(::boost::math::tgamma(static_cast<T>(50.5), ldexp(static_cast<T>(1), -17)), static_cast<T>(4.2904629123519598109157551960589377e63L), tolerance * 10);
      BOOST_CHECK_CLOSE(::boost::math::tgamma(static_cast<T>(164.5), static_cast<T>(0.125)), static_cast<T>(2.5649307433687542701168405519538910e292L), tolerance * 10);
   }
   //
   // Check very large parameters, see: https://github.com/boostorg/math/issues/168
   //
   T max_val = boost::math::tools::max_value<T>();
   T large_val = max_val * 0.99f;
   BOOST_CHECK_EQUAL(::boost::math::tgamma(static_cast<T>(22.25), max_val), 0);
   BOOST_CHECK_EQUAL(::boost::math::tgamma(static_cast<T>(22.25), large_val), 0);
   BOOST_CHECK_EQUAL(::boost::math::tgamma_lower(static_cast<T>(22.25), max_val), boost::math::tgamma(static_cast<T>(22.25)));
   BOOST_CHECK_EQUAL(::boost::math::tgamma_lower(static_cast<T>(22.25), large_val), boost::math::tgamma(static_cast<T>(22.25)));
   BOOST_CHECK_EQUAL(::boost::math::gamma_q(static_cast<T>(22.25), max_val), 0);
   BOOST_CHECK_EQUAL(::boost::math::gamma_q(static_cast<T>(22.25), large_val), 0);
   BOOST_CHECK_EQUAL(::boost::math::gamma_p(static_cast<T>(22.25), max_val), 1);
   BOOST_CHECK_EQUAL(::boost::math::gamma_p(static_cast<T>(22.25), large_val), 1);
   if (std::numeric_limits<T>::has_infinity)
   {
      BOOST_CHECK_EQUAL(::boost::math::tgamma(static_cast<T>(22.25), std::numeric_limits<T>::infinity()), 0);
      BOOST_CHECK_EQUAL(::boost::math::tgamma_lower(static_cast<T>(22.25), std::numeric_limits<T>::infinity()), boost::math::tgamma(static_cast<T>(22.25)));
      BOOST_CHECK_EQUAL(::boost::math::gamma_q(static_cast<T>(22.25), std::numeric_limits<T>::infinity()), 0);
      BOOST_CHECK_EQUAL(::boost::math::gamma_p(static_cast<T>(22.25), std::numeric_limits<T>::infinity()), 1);
   }
   //
   // Large arguments and small parameters, see https://github.com/boostorg/math/issues/451:
   //
   BOOST_CHECK_EQUAL(::boost::math::gamma_q(static_cast<T>(1770), static_cast<T>(1e-12)), 1);
   BOOST_CHECK_EQUAL(::boost::math::gamma_p(static_cast<T>(1770), static_cast<T>(1e-12)), 0);
}

