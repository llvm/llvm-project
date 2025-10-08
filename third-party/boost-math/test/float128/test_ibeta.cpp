///////////////////////////////////////////////////////////////
//  Copyright 2011 John Maddock. Distributed under the Boost
//  Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_

#include "setup.hpp"
#include "table_type.hpp"

#include <boost/math/special_functions/beta.hpp>
#include "test_ibeta.hpp"

void expected_results()
{
   //
   // Define the max and mean errors expected for
   // various compilers and platforms.
   //
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "[^|]*",                          // platform
      ".*",                             // test type(s)
      "(?i).*small.*",                  // test data group
      ".*", 90, 25);  // test function
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "[^|]*",                          // platform
      ".*",                             // test type(s)
      "(?i).*medium.*",                 // test data group
      ".*", 150, 50);  // test function
   add_expected_result(
      "[^|]*",                          // compiler
      "[^|]*",                          // stdlib
      "[^|]*",                          // platform
      ".*",                             // test type(s)
      "(?i).*large.*",                  // test data group
      ".*", 150000, 5000);                 // test function
   //
   // Finish off by printing out the compiler/stdlib/platform names,
   // we do this to make it easier to mark up expected error rates.
   //
   std::cout << "Tests run with " << BOOST_COMPILER << ", "
      << BOOST_STDLIB << ", " << BOOST_PLATFORM << std::endl;
}

template <class T>
void test(T t, const char* p)
{
   test_beta(t, p);
}


BOOST_AUTO_TEST_CASE( test_main )
{
   expected_results();
   ALL_TESTS
}

