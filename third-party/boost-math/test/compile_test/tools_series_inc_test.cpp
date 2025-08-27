//  Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Basic sanity check that header <boost/math/tools/series.hpp>
// #includes all the files that it needs to.
//
#include <boost/math/tools/series.hpp>
//
// Note this header includes no other headers, this is
// important if this test is to be meaningful:
//
#include "test_compile_result.hpp"

struct Functor
{
   typedef double result_type;
   double operator()();
};
#define U double

Functor func;
std::uintmax_t uim = 0;

void compile_and_link_test()
{
   check_result<Functor::result_type>(boost::math::tools::sum_series<Functor>(func, i));
   check_result<Functor::result_type>(boost::math::tools::sum_series<Functor>(func, i, uim));
   check_result<Functor::result_type>(boost::math::tools::sum_series<Functor, U>(func, i, d));
   check_result<Functor::result_type>(boost::math::tools::sum_series<Functor, U>(func, i, uim, d));
   check_result<Functor::result_type>(boost::math::tools::kahan_sum_series<Functor>(func, i));
   check_result<Functor::result_type>(boost::math::tools::kahan_sum_series<Functor>(func, i, uim));
}

