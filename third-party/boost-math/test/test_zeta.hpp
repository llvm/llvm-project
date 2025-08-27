// Copyright John Maddock 2006.
// Copyright Paul A. Bristow 2007, 2009
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/math/concepts/real_concept.hpp>
#include <boost/math/special_functions/math_fwd.hpp>
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/tools/stats.hpp>
#include <boost/math/tools/test.hpp>
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
void do_test_zeta(const T& data, const char* type_name, const char* test_name)
{
#if !(defined(ERROR_REPORTING_MODE) && !defined(ZETA_FUNCTION_TO_TEST))
   //
   // test zeta(T) against data:
   //
   using namespace std;
   typedef Real                   value_type;

   std::cout << test_name << " with type " << type_name << std::endl;

   typedef value_type (*pg)(value_type);
#ifdef ZETA_FUNCTION_TO_TEST
   pg funcp = ZETA_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   pg funcp = boost::math::zeta<value_type>;
#else
   pg funcp = boost::math::zeta;
#endif

   boost::math::tools::test_result<value_type> result;
   //
   // test zeta against data:
   //
   result = boost::math::tools::test_hetero<Real>(
      data,
      bind_func<Real>(funcp, 0),
      extract_result<Real>(1));
   handle_test_result(result, data[result.worst()], result.worst(), type_name, "zeta", test_name);
   std::cout << std::endl;
#endif
}
template <class T>
void test_zeta(T, const char* name)
{
   //
   // The actual test data is rather verbose, so it's in a separate file
   //
#include "zeta_data.ipp"
   do_test_zeta<T>(zeta_data, name, "Zeta: Random values greater than 1");
#include "zeta_neg_data.ipp"
   do_test_zeta<T>(zeta_neg_data, name, "Zeta: Random values less than 1");
#include "zeta_1_up_data.ipp"
   do_test_zeta<T>(zeta_1_up_data, name, "Zeta: Values close to and greater than 1");
#include "zeta_1_below_data.ipp"
   do_test_zeta<T>(zeta_1_below_data, name, "Zeta: Values close to and less than 1");

   std::array<std::array<typename table_type<T>::type, 2>, 90> integer_data = { {
      {{ SC_(2.0), SC_(1.6449340668482264364724151666460252) }}, {{ SC_(3.0), SC_(1.2020569031595942853997381615114500) }}, {{ SC_(4.0), SC_(1.0823232337111381915160036965411679) }}, {{ SC_(5.0), SC_(1.0369277551433699263313654864570342) }}, {{ SC_(6.0), SC_(1.0173430619844491397145179297909205) }}, {{ SC_(7.0), SC_(1.0083492773819228268397975498497968) }}, {{ SC_(8.0), SC_(1.0040773561979443393786852385086525) }}, {{ SC_(9.0), SC_(1.0020083928260822144178527692324121) }}, {{ SC_(10.0), SC_(1.0009945751278180853371459589003190) }}, {{ SC_(11.0), SC_(1.0004941886041194645587022825264699) }}, {{ SC_(12.0), SC_(1.0002460865533080482986379980477397) }}, {{ SC_(13.0), SC_(1.0001227133475784891467518365263574) }}, {{ SC_(14.0), SC_(1.0000612481350587048292585451051353) }}, {{ SC_(15.0), SC_(1.0000305882363070204935517285106451) }}, {{ SC_(16.0), SC_(1.0000152822594086518717325714876367) }}, {{ SC_(17.0), SC_(1.0000076371976378997622736002935630) }}, {{ SC_(18.0), SC_(1.0000038172932649998398564616446219) }}, {{ SC_(19.0), SC_(1.0000019082127165539389256569577951) }}, {{ SC_(20.0), SC_(1.0000009539620338727961131520386834) }}, {{ SC_(21.0), SC_(1.0000004769329867878064631167196044) }}, {{ SC_(22.0), SC_(1.0000002384505027277329900036481868) }}, {{ SC_(23.0), SC_(1.0000001192199259653110730677887189) }}, {{ SC_(24.0), SC_(1.0000000596081890512594796124402079) }}, {{ SC_(25.0), SC_(1.0000000298035035146522801860637051) }}, {{ SC_(26.0), SC_(1.0000000149015548283650412346585066) }}, {{ SC_(27.0), SC_(1.0000000074507117898354294919810042) }}, {{ SC_(28.0), SC_(1.0000000037253340247884570548192040) }}, {{ SC_(29.0), SC_(1.0000000018626597235130490064039099) }}, {{ SC_(30.0), SC_(1.0000000009313274324196681828717647) }}, {{ SC_(31.0), SC_(1.0000000004656629065033784072989233) }}, {{ SC_(32.0), SC_(1.0000000002328311833676505492001456) }}, {{ SC_(33.0), SC_(1.0000000001164155017270051977592974) }}, {{ SC_(34.0), SC_(1.0000000000582077208790270088924369) }}, {{ SC_(35.0), SC_(1.0000000000291038504449709968692943) }}, {{ SC_(36.0), SC_(1.0000000000145519218910419842359296) }}, {{ SC_(37.0), SC_(1.0000000000072759598350574810145209) }}, {{ SC_(38.0), SC_(1.0000000000036379795473786511902372) }}, {{ SC_(39.0), SC_(1.0000000000018189896503070659475848) }}, {{ SC_(40.0), SC_(1.0000000000009094947840263889282533) }}, {{ SC_(41.0), SC_(1.0000000000004547473783042154026799) }}, {{ SC_(42.0), SC_(1.0000000000002273736845824652515227) }}, {{ SC_(43.0), SC_(1.0000000000001136868407680227849349) }}, {{ SC_(44.0), SC_(1.0000000000000568434198762758560928) }}, {{ SC_(45.0), SC_(1.0000000000000284217097688930185546) }}, {{ SC_(46.0), SC_(1.0000000000000142108548280316067698) }}, {{ SC_(47.0), SC_(1.0000000000000071054273952108527129) }}, {{ SC_(48.0), SC_(1.0000000000000035527136913371136733) }}, {{ SC_(49.0), SC_(1.0000000000000017763568435791203275) }}, {{ SC_(50.0), SC_(1.0000000000000008881784210930815903) }}, {{ SC_(51.0), SC_(1.0000000000000004440892103143813364) }}, {{ SC_(52.0), SC_(1.0000000000000002220446050798041984) }}, {{ SC_(53.0), SC_(1.0000000000000001110223025141066134) }}, {{ SC_(54.0), SC_(1.0000000000000000555111512484548124) }}, {{ SC_(55.0), SC_(1.0000000000000000277555756213612417) }}, {{ SC_(56.0), SC_(1.0000000000000000138777878097252328) }}, {{ SC_(57.0), SC_(1.0000000000000000069388939045441537) }}, {{ SC_(58.0), SC_(1.0000000000000000034694469521659226) }}, {{ SC_(59.0), SC_(1.0000000000000000017347234760475766) }}, {{ SC_(60.0), SC_(1.0000000000000000008673617380119934) }},
      {{ SC_(-61.0), SC_(-3.3066089876577576725680214670439210e34) }}, {{ SC_(-59.0), SC_(3.5666582095375556109684574608651829e32) }}, {{ SC_(-57.0), SC_(-4.1147288792557978697665486067619336e30) }}, {{ SC_(-55.0), SC_(5.0890659468662289689766332915911925e28) }}, {{ SC_(-53.0), SC_(-6.7645882379292820990945242301798478e26) }}, {{ SC_(-51.0), SC_(9.6899578874635940656497942894654088e24) }}, {{ SC_(-49.0), SC_(-1.5001733492153928733711440151515152e23) }}, {{ SC_(-47.0), SC_(2.5180471921451095697089023320225526e21) }}, {{ SC_(-45.0), SC_(-4.5979888343656503490437943262411348e19) }}, {{ SC_(-43.0), SC_(9.1677436031953307756992753623188406e17) }}, {{ SC_(-41.0), SC_(-2.0040310656516252738108421663238939e16) }}, {{ SC_(-39.0), SC_(4.8241448354850170371581670362158167e14) }}, {{ SC_(-37.0), SC_(-1.2850850499305083333333333333333333e13) }}, {{ SC_(-35.0), SC_(3.8087931125245368811553022079337869e11) }}, {{ SC_(-33.0), SC_(-1.2635724795916666666666666666666667e10) }}, {{ SC_(-31.0), SC_(4.7238486772162990196078431372549020e8) }}, {{ SC_(-29.0), SC_(-2.0052695796688078946143462272494531e7) }}, {{ SC_(-27.0), SC_(974936.82385057471264367816091954023) }}, {{ SC_(-25.0), SC_(-54827.583333333333333333333333333333) }}, {{ SC_(-23.0), SC_(3607.5105463980463980463980463980464) }}, {{ SC_(-21.0), SC_(-281.46014492753623188405797101449275) }}, {{ SC_(-19.0), SC_(26.456212121212121212121212121212121) }}, {{ SC_(-17.0), SC_(-3.0539543302701197438039543302701197) }}, {{ SC_(-15.0), SC_(0.44325980392156862745098039215686275) }}, {{ SC_(-13.0), SC_(-0.083333333333333333333333333333333333) }}, {{ SC_(-11.0), SC_(0.021092796092796092796092796092796093) }}, {{ SC_(-9.0), SC_(-0.0075757575757575757575757575757575758) }}, {{ SC_(-7.0), SC_(0.0041666666666666666666666666666666667) }}, {{ SC_(-5.0), SC_(-0.0039682539682539682539682539682539683) }}, {{ SC_(-3.0), SC_(0.0083333333333333333333333333333333333) }}, {{ SC_(-1.0), SC_(-0.083333333333333333333333333333333333) }}
   } };
   do_test_zeta<T>(integer_data, name, "Zeta: Integer arguments");
}

extern "C" double zetac(double);

template <class T>
void test_spots(T, const char* t)
{
   std::cout << "Testing basic sanity checks for type " << t << std::endl;
   //
   // Basic sanity checks, tolerance is either 5 or 10 epsilon 
   // expressed as a percentage:
   //
   BOOST_MATH_STD_USING
   T tolerance = boost::math::tools::epsilon<T>() * 100 *
      (boost::is_floating_point<T>::value ? 5 : 10);
   // An extra fudge factor for real_concept which has a less accurate tgamma:
   T tolerance_tgamma_extra = std::numeric_limits<T>::is_specialized ? 1 : 10;

   BOOST_CHECK_CLOSE(::boost::math::zeta(static_cast<T>(0.125)), static_cast<T>(-0.63277562349869525529352526763564627152686379131122L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::zeta(static_cast<T>(1023) / static_cast<T>(1024)), static_cast<T>(-1023.4228554489429786541032870895167448906103303056L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::zeta(static_cast<T>(1025) / static_cast<T>(1024)), static_cast<T>(1024.5772867695045940578681624248887776501597556226L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::zeta(static_cast<T>(0.5)), static_cast<T>(-1.46035450880958681288949915251529801246722933101258149054289L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::zeta(static_cast<T>(1.125)), static_cast<T>(8.5862412945105752999607544082693023591996301183069L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::zeta(static_cast<T>(2)), static_cast<T>(1.6449340668482264364724151666460251892189499012068L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::zeta(static_cast<T>(3.5)), static_cast<T>(1.1267338673170566464278124918549842722219969574036L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::zeta(static_cast<T>(4)), static_cast<T>(1.08232323371113819151600369654116790277475095191872690768298L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::zeta(4 + static_cast<T>(1) / 1024), static_cast<T>(1.08225596856391369799036835439238249195298434901488518878804L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::zeta(static_cast<T>(4.5)), static_cast<T>(1.05470751076145426402296728896028011727249383295625173068468L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::zeta(static_cast<T>(6.5)), static_cast<T>(1.01200589988852479610078491680478352908773213619144808841031L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::zeta(static_cast<T>(7.5)), static_cast<T>(1.00582672753652280770224164440459408011782510096320822989663L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::zeta(static_cast<T>(8.125)), static_cast<T>(1.0037305205308161603183307711439385250181080293472L), 2 * tolerance);
   BOOST_CHECK_CLOSE(::boost::math::zeta(static_cast<T>(16.125)), static_cast<T>(1.0000140128224754088474783648500235958510030511915L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::zeta(static_cast<T>(0)), static_cast<T>(-0.5L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::zeta(static_cast<T>(-0.125)), static_cast<T>(-0.39906966894504503550986928301421235400280637468895L), tolerance * tolerance_tgamma_extra);
   BOOST_CHECK_CLOSE(::boost::math::zeta(static_cast<T>(-1)), static_cast<T>(-0.083333333333333333333333333333333333333333333333333L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::zeta(static_cast<T>(-2)), static_cast<T>(0L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::zeta(static_cast<T>(-2.5)), static_cast<T>(0.0085169287778503305423585670283444869362759902200745L), tolerance * 3);
   BOOST_CHECK_CLOSE(::boost::math::zeta(static_cast<T>(-3)), static_cast<T>(0.0083333333333333333333333333333333333333333333333333L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::zeta(static_cast<T>(-4)), static_cast<T>(0), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::zeta(static_cast<T>(-20)), static_cast<T>(0), tolerance * 100);
   BOOST_CHECK_CLOSE(::boost::math::zeta(static_cast<T>(-21)), static_cast<T>(-281.46014492753623188405797101449275362318840579710L), tolerance * 100);
   BOOST_CHECK_CLOSE(::boost::math::zeta(static_cast<T>(-30.125)), static_cast<T>(2.2762941726834511267740045451463455513839970804578e7L), tolerance * 100);
   // Very small values:
   BOOST_CHECK_CLOSE(::boost::math::zeta(ldexp(static_cast<T>(1), -20)), static_cast<T>(-0.500000876368989859479646132126454890645615288202492097957612L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::zeta(ldexp(static_cast<T>(1), -21)), static_cast<T>(-0.500000438184266833093492063649184012943132422189989164545507L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::zeta(ldexp(static_cast<T>(1), -22)), static_cast<T>(-0.500000219092076392425852854644256723571669269957526445270374L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::zeta(ldexp(static_cast<T>(1), -23)), static_cast<T>(-0.500000109546023940187789325464529558825433290921168958481804L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::zeta(ldexp(static_cast<T>(1), -24)), static_cast<T>(-0.500000054773008406088246161057525197302821575823476487961574L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::zeta(ldexp(static_cast<T>(1), -25)), static_cast<T>(-0.500000027386503312042790426817221131071450407798601059264341L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::zeta(ldexp(static_cast<T>(1), -26)), static_cast<T>(-0.500000013693251433271071983943082871935521396740331377486886L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::zeta(ldexp(static_cast<T>(1), -27)), static_cast<T>(-0.500000006846625660947956426350389518286874288247134329498289L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::zeta(ldexp(static_cast<T>(1), -28)), static_cast<T>(-0.500000003423312816552083476988056486473169377162409806781384L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::zeta(ldexp(static_cast<T>(1), -29)), static_cast<T>(-0.500000001711656404795568073849512135664960180586820144333542L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::zeta(ldexp(static_cast<T>(1), -30)), static_cast<T>(-0.500000000855828201527665623188910582717329375986726355164261L), tolerance * 2);
   BOOST_CHECK_CLOSE(::boost::math::zeta(ldexp(static_cast<T>(1), -31)), static_cast<T>(-0.500000000427914100546303208463654361814800355929815322493143L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::zeta(ldexp(static_cast<T>(1), -32)), static_cast<T>(-0.500000000213957050218769203487022003676593508474107873788445L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::zeta(ldexp(static_cast<T>(1), -33)), static_cast<T>(-0.500000000106978525095789001562046589421133388262409441738089L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::zeta(ldexp(static_cast<T>(1), -34)), static_cast<T>(-0.500000000053489262544495600736249301842352101231724731340202L), tolerance);

   BOOST_CHECK_CLOSE(::boost::math::zeta(-ldexp(static_cast<T>(1), -20)), static_cast<T>(-0.499999123632834911086872289657767335473025908373776645822722L), tolerance * tolerance_tgamma_extra);
   BOOST_CHECK_CLOSE(::boost::math::zeta(-ldexp(static_cast<T>(1), -21)), static_cast<T>(-0.499999561816189359548137231641582253243376087534976981434190L), tolerance * tolerance_tgamma_extra);
   BOOST_CHECK_CLOSE(::boost::math::zeta(-ldexp(static_cast<T>(1), -22)), static_cast<T>(-0.499999780908037655734554449793729262345041281451929584703788L), tolerance * tolerance_tgamma_extra);
   BOOST_CHECK_CLOSE(::boost::math::zeta(-ldexp(static_cast<T>(1), -23)), static_cast<T>(-0.499999890454004571852312499433422838864632848598847415933664L), tolerance * tolerance_tgamma_extra);
   BOOST_CHECK_CLOSE(::boost::math::zeta(-ldexp(static_cast<T>(1), -24)), static_cast<T>(-0.499999945226998721921779295091241395945379526155584220813497L), tolerance * tolerance_tgamma_extra);
   BOOST_CHECK_CLOSE(::boost::math::zeta(-ldexp(static_cast<T>(1), -25)), static_cast<T>(-0.499999972613498469959715937215237923104705216198368099221577L), tolerance * tolerance_tgamma_extra);
   BOOST_CHECK_CLOSE(::boost::math::zeta(-ldexp(static_cast<T>(1), -26)), static_cast<T>(-0.499999986306749012229554607064736104475024094525587925697276L), tolerance * tolerance_tgamma_extra);
   BOOST_CHECK_CLOSE(::boost::math::zeta(-ldexp(static_cast<T>(1), -27)), static_cast<T>(-0.499999993153374450427200221401546739119918746163907954406855L), tolerance * tolerance_tgamma_extra);
   BOOST_CHECK_CLOSE(::boost::math::zeta(-ldexp(static_cast<T>(1), -28)), static_cast<T>(-0.499999996576687211291705684949926422460038672790251466963619L), tolerance * tolerance_tgamma_extra);
   BOOST_CHECK_CLOSE(::boost::math::zeta(-ldexp(static_cast<T>(1), -29)), static_cast<T>(-0.499999998288343602165379216634983519354686193860717726606017L), tolerance * tolerance_tgamma_extra);
   BOOST_CHECK_CLOSE(::boost::math::zeta(-ldexp(static_cast<T>(1), -30)), static_cast<T>(-0.499999999144171800212571199432213326524228740247618955829902L), tolerance * tolerance_tgamma_extra);
   BOOST_CHECK_CLOSE(::boost::math::zeta(-ldexp(static_cast<T>(1), -31)), static_cast<T>(-0.499999999572085899888755997191626615213504580792674808876724L), tolerance * tolerance_tgamma_extra);
   BOOST_CHECK_CLOSE(::boost::math::zeta(-ldexp(static_cast<T>(1), -32)), static_cast<T>(-0.499999999786042949889995597926798240562852438685508646794693L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::zeta(-ldexp(static_cast<T>(1), -33)), static_cast<T>(-0.499999999893021474931402198791408471637626205588681812641711L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::zeta(-ldexp(static_cast<T>(1), -34)), static_cast<T>(-0.499999999946510737462302199352114463422268928922372277519378L), tolerance);
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4127 4756)
#endif
   //
   // Very large negative values need special handling in our code, test them here, due to a bug report by Rocco Romeo:
   //
   BOOST_CHECK_EQUAL(::boost::math::zeta(static_cast<T>(-200)), static_cast<T>(0));
   if(std::numeric_limits<T>::max_exponent >= 1024)
   {
      BOOST_CHECK_CLOSE(::boost::math::zeta(static_cast<T>(-171)), static_cast<T>(1.28194898634822427378088228065956967928127061276520385040051e172L), tolerance * 200);
      BOOST_CHECK_CLOSE(::boost::math::zeta(static_cast<T>(-171.5)), static_cast<T>(4.73930233055054501360661283732419615206017226423071857829425e172L), tolerance * 1000);
      BOOST_CHECK_CLOSE(::boost::math::zeta(static_cast<T>(-172.5)), static_cast<T>(-1.30113885243175165293156588942160456456090687128236657847674e174L), tolerance * 100);
      BOOST_CHECK_CLOSE(::boost::math::zeta(static_cast<T>(-173)), static_cast<T>(-9.66241211085609184243169684777934860657838245104636064505158e174L), tolerance * 100);
   }
   if(std::numeric_limits<T>::has_infinity)
   {
      BOOST_CHECK_EQUAL(::boost::math::zeta(static_cast<T>(-10007)), std::numeric_limits<T>::infinity());
      BOOST_CHECK_EQUAL(::boost::math::zeta(static_cast<T>(-10009)), -std::numeric_limits<T>::infinity());
   }
#ifdef _MSC_VER
#pragma warning(pop)
#endif
}

