//  Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Basic sanity check that header <boost/math/tools/test.hpp>
// #includes all the files that it needs to.
//
#include <array>

#ifndef BOOST_MATH_STANDALONE
#include "../../include_private/boost/math/tools/test.hpp"
//
// Note this header includes no other headers, this is
// important if this test is to be meaningful:
//
inline void check_result_imp(boost::math::tools::test_result<double>, boost::math::tools::test_result<double>){}

#include "test_compile_result.hpp"


void compile_and_link_test()
{
   check_result<float>(boost::math::tools::relative_error<float>(f, f));

   #define A std::array<std::array<double, 2>, 2>
   typedef double (*F1)(const std::array<double, 2>&);
   typedef F1 F2;
   A a;
   F1 f1 = 0;
   F2 f2 = 0;

   check_result<boost::math::tools::test_result<
      boost::math::tools::calculate_result_type<A>::value_type> >
      (boost::math::tools::test<A, F1, F2>(a, f1, f2));

}
#endif
