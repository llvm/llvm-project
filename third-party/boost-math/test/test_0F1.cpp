//  (C) Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "test_0F1.hpp"

#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>

void expected_results()
{
   //
   // Define the max and mean errors expected for
   // various compilers and platforms.
   //
   const char* largest_type;
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   if(boost::math::policies::digits<double, boost::math::policies::policy<> >() == boost::math::policies::digits<long double, boost::math::policies::policy<> >())
   {
      largest_type = "(long\\s+)?double|real_concept|cpp_bin_float_quad|dec_40";
   }
   else
   {
      largest_type = "long double|real_concept|cpp_bin_float_quad|dec_40";
   }
#else
   largest_type = "(long\\s+)?double|cpp_bin_float_quad|dec_40";
#endif

   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      largest_type,                  // test type(s)
      "Integer.*",                   // test data group
      ".*", 300, 100);               // test function
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      largest_type,                  // test type(s)
      "Real.*",                   // test data group
      ".*", 2000, 1000);               // test function

   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      largest_type,                  // test type(s)
      "Large.*",                   // test data group
      ".*", 400, 100);               // test function


   //
   // Finish off by printing out the compiler/stdlib/platform names,
   // we do this to make it easier to mark up expected error rates.
   //
   std::cout << "Tests run with " << BOOST_COMPILER << ", " 
      << BOOST_STDLIB << ", " << BOOST_PLATFORM << std::endl;
}

BOOST_AUTO_TEST_CASE( test_main )
{
   expected_results();
   BOOST_MATH_CONTROL_FP;

#if !defined(TEST) || (TEST == 1)
#ifndef BOOST_MATH_BUGGY_LARGE_FLOAT_CONSTANTS
   test_spots(0.0F, "float");
#endif
   test_spots(0.0, "double");
#endif

#if !defined(TEST) || (TEST == 2)
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   test_spots(0.0L, "long double");
#ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
   test_spots(boost::math::concepts::real_concept(0.1), "real_concept");
#endif
#endif
#endif
#if !defined(TEST) || (TEST == 3)
   #if defined(BOOST_MATH_RUN_MP_TESTS)
   test_spots(boost::multiprecision::cpp_bin_float_quad(), "cpp_bin_float_quad");
   #endif
#endif
#if !defined(TEST) || (TEST == 4)
   #if defined(BOOST_MATH_RUN_MP_TESTS)
   typedef boost::multiprecision::number<boost::multiprecision::cpp_dec_float<40> > dec_40;
   test_spots(dec_40(), "dec_40");
   #endif
#endif
}



