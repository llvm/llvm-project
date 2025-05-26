//  Copyright John Maddock 2006-15.
//  Copyright Paul A. Bristow 2007
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "bindings.hpp"
#include "../../test/test_binomial_coeff.hpp"
#include <boost/math/special_functions/binomial.hpp>

BOOST_AUTO_TEST_CASE_EXPECTED_FAILURES(test_main, 10000);

BOOST_AUTO_TEST_CASE(test_main)
{
   BOOST_MATH_CONTROL_FP;

   error_stream_replacer rep;

#ifdef TYPE_TO_TEST

   test_binomial(static_cast<TYPE_TO_TEST>(0), NAME_OF_TYPE_TO_TEST);

#else
   bool test_float = false;
   bool test_double = false;
   bool test_long_double = false;

   if(std::numeric_limits<long double>::digits == std::numeric_limits<double>::digits)
   {
      //
      // Don't bother with long double, it's the same as double:
      //
      if(BOOST_MATH_PROMOTE_FLOAT_POLICY == false)
         test_float = true;
      test_double = true;
   }
   else
   {
      if(BOOST_MATH_PROMOTE_FLOAT_POLICY == false)
         test_float = true;
      if(BOOST_MATH_PROMOTE_DOUBLE_POLICY == false)
         test_double = true;
      test_long_double = true;
   }

#ifdef ALWAYS_TEST_DOUBLE
   test_double = true;
#endif

   if(test_float)
      test_binomial(0.0f, "float");
   if(test_double)
      test_binomial(0.0, "double");
   if(test_long_double)
      test_binomial(0.0L, "long double");
#ifdef BOOST_MATH_USE_FLOAT128
   //test_binomial(0.0Q, "__float128");
#endif


#endif
}

