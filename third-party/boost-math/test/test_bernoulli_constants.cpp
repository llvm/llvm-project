//  (C) Copyright John Maddock 2013.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_TEST_MAIN
#define BOOST_MATH_OVERFLOW_ERROR_POLICY ignore_error

#include <boost/math/tools/config.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/math/concepts/real_concept.hpp>
#include <boost/math/special_functions/bernoulli.hpp>
#include "table_type.hpp"
#include <boost/math/tools/test.hpp>
#include <iostream>
#include <iomanip>

#define SC_(x) static_cast<typename table_type<T>::type>(BOOST_JOIN(x, L))

template <class T>
void test(const char* name)
{
   std::cout << "Testing type " << name << ":\n";

   static const typename table_type<T>::type data[] =
   {
      /* First 50 from 2 to 100 inclusive: */
      /* TABLE[N[BernoulliB[n], 200], {n,2,100,2}] */

      SC_(0.1666666666666666666666666666666666666666),
      SC_(-0.0333333333333333333333333333333333333333),
      SC_(0.0238095238095238095238095238095238095238),
      SC_(-0.0333333333333333333333333333333333333333),
      SC_(0.0757575757575757575757575757575757575757),
      SC_(-0.2531135531135531135531135531135531135531),
      SC_(1.1666666666666666666666666666666666666666),
      SC_(-7.0921568627450980392156862745098039215686),
      SC_(54.9711779448621553884711779448621553884711),
      SC_(-529.1242424242424242424242424242424242424242),
      SC_(6192.1231884057971014492753623188405797101449),
      SC_(-86580.2531135531135531135531135531135531135531),
      SC_(1.4255171666666666666666666666666666666666e6),
      SC_(-2.7298231067816091954022988505747126436781e7),
      SC_(6.0158087390064236838430386817483591677140e8),
      SC_(-1.5116315767092156862745098039215686274509e10),
      SC_(4.2961464306116666666666666666666666666666e11),
      SC_(-1.3711655205088332772159087948561632772159e13),
      SC_(4.8833231897359316666666666666666666666666e14),
      SC_(-1.9296579341940068148632668144863266814486e16),
      SC_(8.4169304757368261500055370985603543743078e17),
      SC_(-4.0338071854059455413076811594202898550724e19),
      SC_(2.1150748638081991605601453900709219858156e21),
      SC_(-1.2086626522296525934602731193708252531781e23),
      SC_(7.5008667460769643668557200757575757575757e24),
      SC_(-5.0387781014810689141378930305220125786163e26),
      SC_(3.6528776484818123335110430842971177944862e28),
      SC_(-2.8498769302450882226269146432910678160919e30),
      SC_(2.3865427499683627644645981919219214971751e32),
      SC_(-2.1399949257225333665810744765191097392674e34),
      SC_(2.0500975723478097569921733095672310251666e36),
      SC_(-2.0938005911346378409095185290027970184709e38),
      SC_(2.2752696488463515559649260352769264581469e40),
      SC_(-2.6257710286239576047303049736158202081449e42),
      SC_(3.2125082102718032518204792304264985243521e44),
      SC_(-4.1598278166794710913917074495262358936689e46),
      SC_(5.6920695482035280023883456219121058644480e48),
      SC_(-8.2183629419784575692290653468617333014550e50),
      SC_(1.2502904327166993016732339829702895524177e53),
      SC_(-2.0015583233248370274925329198813298768724e55),
      SC_(3.3674982915364374233396676903338753016219e57),
      SC_(-5.9470970503135447718660496844051540840579e59),
      SC_(1.1011910323627977559564130790437691604630e62),
      SC_(-2.1355259545253501188658385019041065678973e64),
      SC_(4.3328896986641192419616613059379206218451e66),
      SC_(-9.1885528241669328226200555215501897138960e68),
      SC_(2.0346896776329074493455027990220020065975e71),
      SC_(-4.7003833958035731078575255535006060654596e73),
      SC_(1.1318043445484249270675186257733934267890e76),
      SC_(-2.8382249570693706959264156336481764738284e78),

      /* next 50 from 102 to 200: */
      /* TABLE[N[BernoulliB[n], 200], {n,102,200,2}] */

      SC_(7.4064248979678850629750827140920984176879e80),
      SC_(-2.0096454802756604483465619672715363186867e83),
      SC_(5.6657170050805941445719346030519356961419e85),
      SC_(-1.6584511154136216915823713374319912301494e88),
      SC_(5.0368859950492377419289421915180154812442e90),
      SC_(-1.5861468237658186369363401572966438782740e93),
      SC_(5.1756743617545626984073240682507122561240e95),
      SC_(-1.7488921840217117339690025877618159145141e98),
      SC_(6.1160519994952185255824525264264167780767e100),
      SC_(-2.2122776912707834942288323456712932445573e103),
      SC_(8.2722776798770969854221062459984595731204e105),
      SC_(-3.1958925111415709583591634369180814873526e108),
      SC_(1.2750082223387792982310024302926679866957e111),
      SC_(-5.2500923086774133899402824624565175446919e113),
      SC_(2.2301817894241625209869298198838728143738e116),
      SC_(-9.7684521930955204438633513398980239301166e118),
      SC_(4.4098361978452954272272622874813169191875e121),
      SC_(-2.0508570886464088839729337727583015486456e124),
      SC_(9.8214433279791277107572969602097521041491e126),
      SC_(-4.8412600798208880508789196709963412761130e129),
      SC_(2.4553088801480982609783467404088690399673e132),
      SC_(-1.2806926804084747548782513278601785721811e135),
      SC_(6.8676167104668581192101888598464400436092e137),
      SC_(-3.7846468581969104694978995416379556814489e140),
      SC_(2.1426101250665291550871323135148272096660e143),
      SC_(-1.2456727137183695007019642961637607219458e146),
      SC_(7.4345787551000152543679668394052061311780e148),
      SC_(-4.5535795304641704894063333223321274876772e151),
      SC_(2.8612112816858868345363847251017232522918e154),
      SC_(-1.8437723552033869727688202653628785487541e157),
      SC_(1.2181154536221046699501316506599521355817e160),
      SC_(-8.2482187185314121548481845729689344730141e162),
      SC_(5.7225877937832943329651649814297861591868e165),
      SC_(-4.0668530525059104726767969383115865560219e168),
      SC_(2.9596092064642050062875269581585187042637e171),
      SC_(-2.2049522565189457509031175227344598483637e174),
      SC_(1.6812597072889599805831152515136066575446e177),
      SC_(-1.3116736213556957648645280635581715300443e180),
      SC_(1.0467894009478038082183285392982308964382e183),
      SC_(-8.5432893578833707718598254629908277459327e185),
      SC_(7.1287821322486542352288406677143822472124e188),
      SC_(-6.0802931455535899300084711868647745846198e191),
      SC_(5.2996776424849923930094291004324726622848e194),
      SC_(-4.7194259168745862644364622901337991110376e197),
      SC_(4.2928413791402981089416829654107466904552e200),
      SC_(-3.9876744968232207443447765554293879510665e203),
      SC_(3.7819780419358882713894418116139332789822e206),
      SC_(-3.6614233683681191243685808215119734875519e209),
      SC_(3.6176090272372862348855460929891408947754e212),
      SC_(-3.6470772645191354362138308865549944904868e215),
   };

   T tol = boost::math::tools::epsilon<T>() * 20;
   for(unsigned i = 1; i <= 100; ++i)
   {
      T b2n = boost::math::bernoulli_b2n<T>(i);
      T t2n = boost::math::tangent_t2n<T>(i);
      if((boost::math::isinf)(b2n))
      {
         if(!(boost::math::isinf)(data[i - 1]))
         {
            std::cout << "When calculating B2N(" << i << ")\n";
            BOOST_ERROR("Got an infinity when one wasn't expected");
         }
         else if((b2n > 0) != (data[i - 1] > 0))
         {
            std::cout << "When calculating B2N(" << i << ")\n";
            BOOST_ERROR("Sign of infinity was incorrect");
         }
      }
      else if((boost::math::isnan)(b2n))
      {
         std::cout << "When calculating B2N(" << i << ")\n";
         BOOST_ERROR("Result of B2n was a NaN, and that should never happen!");
      }
      else
      {
         BOOST_CHECK_CLOSE_FRACTION(b2n, data[i - 1], tol);
      }
      if(i <= boost::math::max_bernoulli_b2n<T>::value)
      {
         BOOST_CHECK_EQUAL(b2n, boost::math::unchecked_bernoulli_b2n<T>(i));
      }
      if((boost::math::isfinite)(t2n) && (t2n < boost::math::tools::max_value<T>()))
      {
         T p = ldexp(T(1), 2 * i);
         int s = i & 1 ? 1 : -1;
         p = t2n / (s * p * (p - 1));
         p *= 2 * i;
         BOOST_CHECK_CLOSE_FRACTION(p, b2n, tol);
      }
   }
   //
   // Test consistency of array interface:
   //
   T bn[boost::math::max_bernoulli_b2n<T>::value + 20];
   boost::math::bernoulli_b2n<T>(0, boost::math::max_bernoulli_b2n<T>::value + 20, bn);

   for(unsigned i = 0; i < boost::math::max_bernoulli_b2n<T>::value + 20; ++i)
   {
      BOOST_CHECK_EQUAL(bn[i], boost::math::bernoulli_b2n<T>(i));
   }

   boost::math::tangent_t2n<T>(0, boost::math::max_bernoulli_b2n<T>::value + 20, bn);

   for(unsigned i = 0; i < boost::math::max_bernoulli_b2n<T>::value + 20; ++i)
   {
      BOOST_CHECK_EQUAL(bn[i], boost::math::tangent_t2n<T>(i));
   }
   //
   // Some spot tests for things that should throw exceptions:
   //
   static unsigned overflow_index = boost::is_same<T, boost::math::concepts::real_concept>::value ?
      boost::math::max_bernoulli_b2n<long double>::value + 5 : boost::math::max_bernoulli_b2n<T>::value + 5;
#ifndef BOOST_NO_EXCEPTIONS
   BOOST_MATH_CHECK_THROW(boost::math::bernoulli_b2n<T>(overflow_index, boost::math::policies::make_policy(boost::math::policies::overflow_error<boost::math::policies::throw_on_error>())), std::overflow_error);
   BOOST_MATH_CHECK_THROW(boost::math::tangent_t2n<T>(overflow_index, boost::math::policies::make_policy(boost::math::policies::overflow_error<boost::math::policies::throw_on_error>())), std::overflow_error);
#endif
   BOOST_MATH_CHECK_THROW(boost::math::bernoulli_b2n<T>(-1), std::domain_error);
   BOOST_MATH_CHECK_THROW(boost::math::tangent_t2n<T>(-1), std::domain_error);
   std::vector<T> v;
   BOOST_MATH_CHECK_THROW(boost::math::bernoulli_b2n<T>(-1, 5, std::back_inserter(v)), std::domain_error);
   BOOST_MATH_CHECK_THROW(boost::math::tangent_t2n<T>(-1, 5, std::back_inserter(v)), std::domain_error);
}

void test_real_concept_extra()
{
#if !defined(BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS) && !defined(BOOST_MATH_NO_REAL_CONCEPT_TESTS)
   boost::math::concepts::real_concept tol = boost::math::tools::epsilon<boost::math::concepts::real_concept>() * 20;
   for(unsigned i = 0; i <= boost::math::max_bernoulli_b2n<long double>::value; ++i)
   {
      BOOST_CHECK_CLOSE_FRACTION(static_cast<boost::math::concepts::real_concept>(boost::math::bernoulli_b2n<long double>(i)), boost::math::bernoulli_b2n<boost::math::concepts::real_concept>(i), tol);
   }
   for(unsigned i = 1; i < 500; ++i)
   {
      boost::math::concepts::real_concept r = boost::math::bernoulli_b2n<long double>(i + boost::math::max_bernoulli_b2n<long double>::value);
      if((i + boost::math::max_bernoulli_b2n<long double>::value) & 1)
      {
         BOOST_CHECK(r >= boost::math::tools::max_value<boost::math::concepts::real_concept>());
      }
      else
      {
         BOOST_CHECK(r <= -boost::math::tools::max_value<boost::math::concepts::real_concept>());
      }
   }
#endif
}


BOOST_AUTO_TEST_CASE( test_main )
{
   test<float>("float");
   test<double>("double");
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   test<long double>("long double");
#ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
   test<boost::math::concepts::real_concept>("real_concept");
   test_real_concept_extra();
#endif
#endif
}


