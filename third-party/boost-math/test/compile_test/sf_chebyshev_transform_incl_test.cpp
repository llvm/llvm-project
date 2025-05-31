//  Copyright Matt Borland 2021
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Basic sanity check that header <boost/math/special_functions/chebyshev_transform.hpp>
// #includes all the files that it needs to.
//
#if __has_include(<fftw3.h>)

#include <boost/math/special_functions/chebyshev_transform.hpp>
//
// Note this header includes no other headers, this is
// important if this test is to be meaningful:
//
#include "test_compile_result.hpp"

void compile_and_link_test()
{   
    auto f = [](double x) { return x; };
    boost::math::chebyshev_transform<double> test(f, 0.0, 1.0);
    check_result<double>(test(1.0));
}

#else
void compile_and_link_test() {}
#endif // __has_include
