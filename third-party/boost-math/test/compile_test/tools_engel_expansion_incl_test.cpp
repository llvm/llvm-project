//  Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Basic sanity check that header
// #includes all the files that it needs to.
//
#include <boost/math/tools/engel_expansion.hpp>
//
// Note this header includes no other headers, this is
// important if this test is to be meaningful:
//
#include "test_compile_result.hpp"

void compile_and_link_test()
{
    boost::math::tools::engel_expansion<float> f_test(1.0f);
    check_result<int64_t>(f_test.digits().front());

    boost::math::tools::engel_expansion<double> d_test(1.0);
    check_result<int64_t>(d_test.digits().front());

#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    boost::math::tools::engel_expansion<long double> ld_test(1.0l);
    check_result<int64_t>(ld_test.digits().front());
#endif
}
