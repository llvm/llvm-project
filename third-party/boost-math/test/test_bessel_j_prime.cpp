//  Copyright (c) 2013 Anton Bikineev
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "pch_light.hpp"

#include "test_bessel_j_prime.hpp"

//
// DESCRIPTION:
// ~~~~~~~~~~~~
//
// This file tests the bessel functions derivatives. There are two sets of tests, spot
// tests which compare our results with selected values computed
// using the online special function calculator at 
// functions.wolfram.com, while the bulk of the accuracy tests
// use values generated with Boost.Multiprecision at 50 precision
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
   const char* largest_type;
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   if(boost::math::policies::digits<double, boost::math::policies::policy<> >() == boost::math::policies::digits<long double, boost::math::policies::policy<> >())
   {
      largest_type = "(long\\s+)?double|real_concept";
   }
   else
   {
      largest_type = "long double|real_concept";
   }
#else
   largest_type = "(long\\s+)?double";
#endif
   //
   // HP-UX specific rates:
   //
   // Error rate for double precision are limited by the accuracy of
   // the approximations use, which bracket rather than preserve the root.
   //
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      "HP-UX",                          // platform
      largest_type,                      // test type(s)
      ".*J0'.*Tricky.*",              // test data group
      ".*", 80000000000LL, 80000000000LL);         // test function
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      "HP-UX",                          // platform
      largest_type,                      // test type(s)
      ".*J1'.*Tricky.*",              // test data group
      ".*", 3000000, 2000000);         // test function
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      "HP-UX",                          // platform
      "double",                      // test type(s)
      ".*Tricky.*",              // test data group
      ".*", 100000, 100000);         // test function
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      "HP-UX",                          // platform
      largest_type,                      // test type(s)
      ".*J'.*Tricky.*",              // test data group
      ".*", 3000, 500);         // test function
   //
   // HP Tru64:
   //
   add_expected_result(
      ".*Tru64.*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      "double",                      // test type(s)
      ".*Tricky.*",              // test data group
      ".*", 100000, 100000);         // test function
   add_expected_result(
      ".*Tru64.*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      largest_type,                      // test type(s)
      ".*Tricky large.*",              // test data group
      ".*", 3000, 1000);         // test function
   //
   // Solaris specific rates:
   //
   // Error rate for double precision are limited by the accuracy of
   // the approximations use, which bracket rather than preserve the root.
   //
   add_expected_result(
      ".*",                              // compiler
      ".*",                              // stdlib
      "Sun Solaris",                     // platform
      largest_type,                      // test type(s)
      "Bessel J': Random Data.*Tricky.*", // test data group
      ".*", 3000, 500);                  // test function
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      "Sun Solaris",                 // platform
      "double",                      // test type(s)
      ".*Tricky.*",                  // test data group
      ".*", 200000, 100000);         // test function
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      "Sun Solaris",                 // platform
      largest_type,                  // test type(s)
      ".*J'.*tricky.*",               // test data group
      ".*", 400000000, 200000000);    // test function
   //
   // Mac OS X:
   //
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      "Mac OS",                          // platform
      largest_type,                  // test type(s)
      ".*J0'.*Tricky.*",              // test data group
      ".*", 400000000, 400000000);   // test function
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      "Mac OS",                          // platform
      largest_type,                  // test type(s)
      ".*J1'.*Tricky.*",              // test data group
      ".*", 3000000, 2000000);       // test function
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      "Mac OS",                          // platform
      largest_type,                      // test type(s)
      "Bessel JN'.*",              // test data group
      ".*", 60000, 20000);         // test function
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      "Mac OS",                          // platform
      largest_type,                      // test type(s)
      "Bessel J':.*",              // test data group
      ".*", 60000, 20000);         // test function



   //
   // Linux specific results:
   //
   // sin and cos appear to have only double precision for large
   // arguments on some linux distros:
   //
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      "linux",                          // platform
      largest_type,                      // test type(s)
      ".*J':.*",              // test data group
      ".*", 60000, 30000);         // test function


#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   if((std::numeric_limits<double>::digits != std::numeric_limits<long double>::digits)
      && (std::numeric_limits<long double>::digits < 90))
   {
      // some errors spill over into type double as well:
      add_expected_result(
         ".*",                          // compiler
         ".*",                          // stdlib
         ".*",                          // platform
         "double",                      // test type(s)
         ".*J0'.*Tricky.*",              // test data group
         ".*", 400000, 400000);         // test function
      add_expected_result(
         ".*",                          // compiler
         ".*",                          // stdlib
         ".*",                          // platform
         "double",                      // test type(s)
         ".*J1'.*Tricky.*",              // test data group
         ".*", 5000, 5000);             // test function
      add_expected_result(
         ".*",                          // compiler
         ".*",                          // stdlib
         ".*",                          // platform
         "double",                      // test type(s)
         ".*(JN'|j').*|.*Tricky.*",       // test data group
         ".*", 50, 50);                 // test function
      add_expected_result(
         ".*",                          // compiler
         ".*",                          // stdlib
         ".*",                          // platform
         "double",                      // test type(s)
         ".*",                          // test data group
         ".*", 30, 30);                 // test function
      //
      // and we have a few cases with higher limits as well:
      //
      add_expected_result(
         ".*",                          // compiler
         ".*",                          // stdlib
         ".*",                          // platform
         largest_type,                  // test type(s)
         ".*J0'.*Tricky.*",              // test data group
         ".*", 400000000, 400000000);   // test function
      add_expected_result(
         ".*",                          // compiler
         ".*",                          // stdlib
         ".*",                          // platform
         largest_type,                  // test type(s)
         ".*J1'.*Tricky.*",              // test data group
         ".*", 5000000, 5000000);       // test function
      add_expected_result(
         ".*",                          // compiler
         ".*",                          // stdlib
         ".*",                          // platform
         largest_type,                  // test type(s)
         ".*(JN'|j').*|.*Tricky.*",       // test data group
         ".*", 60000, 40000);               // test function
   }
#endif
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      largest_type,                  // test type(s)
      ".*J0'.*Tricky.*",              // test data group
      ".*", 400000000, 400000000);   // test function
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      largest_type,                  // test type(s)
      ".*J1'.*Tricky.*",              // test data group
      ".*", 5000000, 5000000);       // test function
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      largest_type,                  // test type(s)
      "Bessel j':.*|Bessel JN': Mathworld.*|.*Tricky.*",       // test data group
      ".*", 1500, 700);               // test function
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      largest_type,                  // test type(s)
      ".*",                          // test data group
      ".*", 40, 20);                 // test function
   //
   // One set of float tests has inexact input values, so there is a slight error:
   //
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      "float|double",                // test type(s)
      "Bessel J': Mathworld Data",    // test data group
      ".*", 30, 20);                 // test function
   //
   // Finish off by printing out the compiler/stdlib/platform names,
   // we do this to make it easier to mark up expected error rates.
   //
   std::cout << "Tests run with " << BOOST_COMPILER << ", " 
      << BOOST_STDLIB << ", " << BOOST_PLATFORM << std::endl;
}

BOOST_AUTO_TEST_CASE( test_main )
{
#ifdef TEST_GSL
   gsl_set_error_handler_off();
#endif
   expected_results();
   BOOST_MATH_CONTROL_FP;

   test_bessel_prime(0.1F, "float");
   test_bessel_prime(0.1, "double");
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   test_bessel_prime(0.1L, "long double");
#ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
   test_bessel_prime(boost::math::concepts::real_concept(0.1), "real_concept");
#endif
#else
   std::cout << "<note>The long double tests have been disabled on this platform "
      "either because the long double overloads of the usual math functions are "
      "not available at all, or because they are too inaccurate for these tests "
      "to pass.</note>" << std::endl;
#endif
}




