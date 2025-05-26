//  (C) Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pch_light.hpp>
#include "test_digamma.hpp"

//
// DESCRIPTION:
// ~~~~~~~~~~~~
//
// This file tests the digamma function.  There are two sets of tests, spot
// tests which compare our results with selected values computed
// using the online special function calculator at 
// functions.wolfram.com, while the bulk of the accuracy tests
// use values generated with NTL::RR at 1000-bit precision
// and our generic versions of these functions.
//
// Note that when this file is first run on a new platform many of
// these tests will fail: the default accuracy is 1 epsilon which
// is too tight for most platforms.  In this situation you will 
// need to cast a human eye over the error rates reported and make
// a judgement as to whether they are acceptable.  Either way please
// report the results to the Boost mailing list.  Acceptable rates of
// error are marked up below as a series of regular expressions that
// identify the compiler/stdlib/platform/data-type/test-data/test-function
// along with the maximum expected peek and RMS mean errors for that
// test.
//

void expected_results()
{
   //
   // Define the max and mean errors expected for
   // various compilers and platforms.
   //
   add_expected_result(
      ".*",                     // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      ".*",                          // test type(s)
      ".*Negative.*",                    // test data group
      ".*", 300, 40);                // test function
   add_expected_result(
      ".*",                     // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      "real_concept",                // test type(s)
      ".*Near the Positive Root.*",  // test data group
      ".*", 25000, 3000);            // test function
   add_expected_result(
      ".*",                     // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      "real_concept",                // test type(s)
      ".*Half.*",                    // test data group
      ".*", 15, 10);                 // test function
   if (std::numeric_limits<long double>::digits > 100)
   {
      add_expected_result(
         ".*",                     // compiler
         ".*",                          // stdlib
         ".*",                          // platform
         "real_concept",                // test type(s)
         ".*Near Zero.*",                    // test data group
         ".*", 15, 10);                 // test function
   }
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      ".*",                          // test type(s)
      ".*",                          // test data group
      ".*", 3, 3);                   // test function
   //
   // Finish off by printing out the compiler/stdlib/platform names,
   // we do this to make it easier to mark up expected error rates.
   //
   std::cout << "Tests run with " << BOOST_COMPILER << ", " 
      << BOOST_STDLIB << ", " << BOOST_PLATFORM << std::endl;
}

template <class T>
void test_spots(T, const char* t)
{
   std::cout << "Testing basic sanity checks for type " << t << std::endl;
   //
   // Basic sanity checks, tolerance is 3 epsilon expressed as a percentage:
   //
   T tolerance = boost::math::tools::epsilon<T>() * 300;
   //
   // Special tolerance (200eps) for when we're very near the root,
   // and T has more than 64-bits in it's mantissa:
   //
   BOOST_CHECK_CLOSE(::boost::math::digamma(static_cast<T>(0.125)), static_cast<T>(-8.3884926632958548678027429230863430000514460424495L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::digamma(static_cast<T>(0.5)), static_cast<T>(-1.9635100260214234794409763329987555671931596046604L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::digamma(static_cast<T>(1)), static_cast<T>(-0.57721566490153286060651209008240243104215933593992L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::digamma(static_cast<T>(1.5)), static_cast<T>(0.036489973978576520559023667001244432806840395339566L), tolerance * 40);
   BOOST_CHECK_CLOSE(::boost::math::digamma(static_cast<T>(1.5) - static_cast<T>(1)/32), static_cast<T>(0.00686541147073577672813890866512415766586241385896200579891429L), tolerance * 200);
   BOOST_CHECK_CLOSE(::boost::math::digamma(static_cast<T>(2)), static_cast<T>(0.42278433509846713939348790991759756895784066406008L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::digamma(static_cast<T>(8)), static_cast<T>(2.0156414779556099965363450527747404261006978069172L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::digamma(static_cast<T>(12)), static_cast<T>(2.4426616799758120167383652547949424463027180089374L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::digamma(static_cast<T>(22)), static_cast<T>(3.0681430398611966699248760264450329818421699570581L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::digamma(static_cast<T>(50)), static_cast<T>(3.9019896734278921969539597028823666609284424880275L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::digamma(static_cast<T>(500)), static_cast<T>(6.2136077650889917423827750552855712637776544784569L), tolerance);
   //
   // negative values:
   //
   BOOST_CHECK_CLOSE(::boost::math::digamma(static_cast<T>(-0.125)), static_cast<T>(7.1959829284523046176757814502538535827603450463013L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::digamma(static_cast<T>(-10.125)), static_cast<T>(9.9480538258660761287008034071425343357982429855241L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::digamma(static_cast<T>(-10.875)), static_cast<T>(-5.1527360383841562620205965901515879492020193154231L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::digamma(static_cast<T>(-1.5)), static_cast<T>(0.70315664064524318722569033366791109947350706200623L), tolerance);
}

BOOST_AUTO_TEST_CASE( test_main )
{
   BOOST_MATH_CONTROL_FP;
   test_spots(0.0F, "float");
   test_spots(0.0, "double");
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   test_spots(0.0L, "long double");
#ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
   test_spots(boost::math::concepts::real_concept(0.1), "real_concept");
#endif
#endif

   expected_results();

   test_digamma(0.1F, "float");
   test_digamma(0.1, "double");
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   test_digamma(0.1L, "long double");
#ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
   test_digamma(boost::math::concepts::real_concept(0.1), "real_concept");
#endif
#else
   std::cout << "<note>The long double tests have been disabled on this platform "
      "either because the long double overloads of the usual math functions are "
      "not available at all, or because they are too inaccurate for these tests "
      "to pass.</note>" << std::endl;
#endif
}


