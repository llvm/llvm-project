//  Copyright John Maddock 2025.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>

#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/beta.hpp>

#include <boost/multiprecision/cpp_bin_float.hpp>


template <class T, class Policy>
T fake_lgamma(T x, const Policy& pol)
{
   typedef typename boost::math::lanczos::lanczos<T, Policy>::type lanczos_type;
   T xm1 = static_cast<T>(x - 1);
   T xm2 = static_cast<T>(x - 2);
   return boost::math::detail::lgamma_small_imp(x, xm1, xm2, boost::math::integral_constant<int, 0>(), pol, lanczos_type());
}

template <class T, class Policy>
T fake_beta(T x, T y, const Policy& pol)
{
   typedef typename boost::math::lanczos::lanczos<T, Policy>::type lanczos_type;
   return boost::math::detail::beta_imp(x, y, lanczos_type(), pol);
}

template <class T, class Policy>
void test_lanczos()
{
   static const T gamma_values[5][2] = {
      { static_cast<T>(0.9990234375), BOOST_MATH_BIG_CONSTANT(T, 1000, 0.0005644719118551233842574575277837954032273381467111560151573964333623759876693326772975355235763579669851459698347791150664129556238023435782509) },
      { static_cast<T>(1.0009765625), BOOST_MATH_BIG_CONSTANT(T, 1000, -0.00056290317999120463170214992168514502772879541390257627479851327242521375496316078955498047974490391404458478547041900753228681266105375203395) },
      { static_cast<T>(1.9990234375), BOOST_MATH_BIG_CONSTANT(T, 1000, -0.00041256773597148940171061762396967043263766321508278648166258325752450997988091121880044028815700667736697239273636914229467280307848301056348) },
      { static_cast<T>(2.0009765625), BOOST_MATH_BIG_CONSTANT(T, 1000, 0.0004131827930642542642586749863320416448830389198819599629600850019751583690627183743966471144142533038637004476283413018111524526428991161605850) },
      { static_cast<T>(15.5), BOOST_MATH_BIG_CONSTANT(T, 1000, 26.536914491115613623952954502438732190637095031219293570786654851418392173706059728481291108517001108710291135377351053352153160697114162788268) },
   };


   T tolerance = boost::math::tools::epsilon<T>() * 10000; // tolerance as a PERSENTAGE

   for (unsigned i = 0; i < 5; ++i)
   {
      BOOST_CHECK_CLOSE(fake_lgamma(gamma_values[i][0], Policy()), gamma_values[i][1], tolerance);
   }

   static const T beta_values[3][3] = {
      { static_cast<T>(3.5), static_cast<T>(5.5), BOOST_MATH_BIG_CONSTANT(T, 1000, 0.0043143209659283659585821213454460946590842475272180103132285310868485027552148255901569081768919454363768513150969758239477301943185356506478162) },
      { static_cast<T>(3.5), static_cast<T>(7.5), BOOST_MATH_BIG_CONSTANT(T, 1000, 0.0017137441614659898113256759788855320451362427677560429855324442928314885944325557205345496369320783261163603834968542856236817160765294390073270) },
      { static_cast<T>(3.5), static_cast<T>(9.5), BOOST_MATH_BIG_CONSTANT(T, 1000, 0.0008276605325261882611516048761663080899805717912458162146037373005152075598111774786672540860183332824993785943024580356705281015142329677024022) },
   };

   for (unsigned i = 0; i < 3; ++i)
   {
      BOOST_CHECK_CLOSE(fake_beta(beta_values[i][0], beta_values[i][1], Policy()), beta_values[i][2], tolerance);
   }
}


int main()
{
   test_lanczos<float, boost::math::policies::policy<boost::math::policies::promote_float<false>>>();
   test_lanczos<double, boost::math::policies::policy<boost::math::policies::promote_double<false>>>();
   test_lanczos<long double, boost::math::policies::policy<>>();

   test_lanczos<boost::multiprecision::cpp_bin_float_double, boost::math::policies::policy<>>();
   test_lanczos<boost::multiprecision::cpp_bin_float_double_extended, boost::math::policies::policy<>>();
   test_lanczos<boost::multiprecision::cpp_bin_float_quad, boost::math::policies::policy<>>();
   test_lanczos<boost::multiprecision::number<boost::multiprecision::cpp_bin_float<130, boost::multiprecision::digit_base_2>>, boost::math::policies::policy<>>();
   test_lanczos<boost::multiprecision::number<boost::multiprecision::cpp_bin_float<160, boost::multiprecision::digit_base_2>>, boost::math::policies::policy<>>();
   test_lanczos<boost::multiprecision::number<boost::multiprecision::cpp_bin_float<200, boost::multiprecision::digit_base_2>>, boost::math::policies::policy<>>();
   test_lanczos<boost::multiprecision::number<boost::multiprecision::cpp_bin_float<230, boost::multiprecision::digit_base_2>>, boost::math::policies::policy<>>();
   test_lanczos<boost::multiprecision::number<boost::multiprecision::cpp_bin_float<260, boost::multiprecision::digit_base_2>>, boost::math::policies::policy<>>();
}
