//  (C) Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "test_1F1.hpp"

#include <boost/multiprecision/cpp_bin_float.hpp>

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
      largest_type = "(long\\s+)?double|real_concept|cpp_bin_float_quad|dec_40|cpp_bin_float_double_extended";
   }
   else
   {
      largest_type = "long double|real_concept|cpp_bin_float_quad|dec_40|cpp_bin_float_double_extended";
   }
#else
   largest_type = "(long\\s+)?double";
#endif

   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      "cpp_bin_float_quad|cpp_bin_float_double_extended",          // test type(s)
      "Integer a values",            // test data group
      ".*", 25000, 800);             // test function
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      "cpp_bin_float_quad|cpp_bin_float_double_extended",          // test type(s)
      "Large.*",                     // test data group
      ".*", 500000, 22000);          // test function
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      "cpp_bin_float_quad|cpp_bin_float_double_extended",          // test type(s)
      "Small.*",                     // test data group
      ".*", 2000, 200);              // test function

   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      "dec_40",                      // test type(s)
      "Integer a values",            // test data group
      ".*", 12000, 800);             // test function
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      "dec_40",                      // test type(s)
      "Large.*",                     // test data group
      ".*", 20000000L, 650000L);     // test function
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      "dec_40",                      // test type(s)
      "Small.*",                     // test data group
      ".*", 1000, 300);              // test function

#if (LDBL_MANT_DIG < DBL_MANT_DIG * 2) && (LDBL_MANT_DIG != DBL_MANT_DIG)
   //
   // long double has only a little extra precision and errors may creep
   // into the double results:
   //
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      "double",                      // test type(s)
      "Integer a values",            // test data group
      ".*", 5, 3);                   // test function

   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      "double",                      // test type(s)
      "Small.*",                     // test data group
      ".*", 5, 3);                   // test function

   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      "double",                      // test type(s)
      "Large.*",                     // test data group
      ".*", 40, 20);                 // test function
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      "double",                      // test type(s)
      "Bug.*",                     // test data group
      ".*", 300, 50);                 // test function
#elif(LDBL_MANT_DIG != DBL_MANT_DIG)
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      "double",                      // test type(s)
      ".*double limited precision.*", // test data group
      ".*", 10, 5);                 // test function

#endif

   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      largest_type,                  // test type(s)
      "Integer a values",                   // test data group
      ".*", 16000, 600);               // test function

   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      largest_type,                  // test type(s)
      "Small.*",                   // test data group
      ".*", 2000, 200);               // test function

   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      largest_type,                  // test type(s)
      "Large.*",                     // test data group
      ".*", 400000, 9000);           // test function
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      "dec_40",                      // test type(s)
      "Bug cases.*",                 // test data group
      ".*", 2200000, 430000);        // test function
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      largest_type,                  // test type(s)
      "Bug cases.*",                 // test data group
      ".*", 1500000, 430000);        // test function
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      largest_type,                  // test type(s)
      ".*negative.*",                // test data group
      ".*", 200, 100);               // test function

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
   test_hypergeometric_mellin_transform<double>();
   test_hypergeometric_laplace_transform<double>();
#endif

#ifndef BOOST_MATH_BUGGY_LARGE_FLOAT_CONSTANTS
#if !defined(TEST) || (TEST == 2)
   test_spots(0.0F, "float");
#endif
#endif
#if !defined(TEST) || (TEST == 3)
   test_spots(0.0, "double");
#endif
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
#if (!defined(TEST) || (TEST == 4)) && (DBL_MAX_EXP != LDBL_MAX_EXP)
   test_spots(0.0L, "long double");
#endif
#ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
#if !defined(TEST) || (TEST == 5)
   test_spots(boost::math::concepts::real_concept(0.1), "real_concept");
#endif
#endif
#endif
   //
   // These next 2 tests take effectively "forever" to compile with clang:
   //
#if (!defined(TEST) || (TEST == 6)) && !defined(__clang__)
   #if defined(BOOST_MATH_RUN_MP_TESTS)
   test_spots(boost::multiprecision::cpp_bin_float_quad(), "cpp_bin_float_quad");
   #endif
#endif
#if (!defined(TEST) || (TEST == 7)) && !defined(__clang__)
   #if defined(BOOST_MATH_RUN_MP_TESTS)
   typedef boost::multiprecision::number<boost::multiprecision::cpp_bin_float<40> > dec_40;
   test_spots(dec_40(), "dec_40");
   #endif
#endif
}

