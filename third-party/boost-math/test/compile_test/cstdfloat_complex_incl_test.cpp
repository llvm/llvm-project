//  Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Basic sanity check that header
// #includes all the files that it needs to.
//
#include <boost/math/cstdfloat/cstdfloat_complex.hpp>
//
// Note this header includes no other headers, this is
// important if this test is to be meaningful:
//
#include "test_compile_result.hpp"

void compile_and_link_test()
{
    #ifdef BOOST_CSTDFLOAT_EXTENDED_COMPLEX_FLOAT_TYPE
    complex<BOOST_CSTDFLOAT_EXTENDED_COMPLEX_FLOAT_TYPE> test(0);

    check_result<BOOST_CSTDFLOAT_EXTENDED_COMPLEX_FLOAT_TYPE>(real(test));
    check_result<BOOST_CSTDFLOAT_EXTENDED_COMPLEX_FLOAT_TYPE>(imag(test));
    check_result<BOOST_CSTDFLOAT_EXTENDED_COMPLEX_FLOAT_TYPE>(abs(test));
    check_result<BOOST_CSTDFLOAT_EXTENDED_COMPLEX_FLOAT_TYPE>(norm(test));
    check_result<BOOST_CSTDFLOAT_EXTENDED_COMPLEX_FLOAT_TYPE>(conj(test));
    check_result<BOOST_CSTDFLOAT_EXTENDED_COMPLEX_FLOAT_TYPE>(proj(test));
    check_result<BOOST_CSTDFLOAT_EXTENDED_COMPLEX_FLOAT_TYPE>(polar(test));
    check_result<BOOST_CSTDFLOAT_EXTENDED_COMPLEX_FLOAT_TYPE>(sqrt(test));
    check_result<BOOST_CSTDFLOAT_EXTENDED_COMPLEX_FLOAT_TYPE>(sin(test));
    check_result<BOOST_CSTDFLOAT_EXTENDED_COMPLEX_FLOAT_TYPE>(cos(test));
    check_result<BOOST_CSTDFLOAT_EXTENDED_COMPLEX_FLOAT_TYPE>(tan(test));
    check_result<BOOST_CSTDFLOAT_EXTENDED_COMPLEX_FLOAT_TYPE>(asin(test));
    check_result<BOOST_CSTDFLOAT_EXTENDED_COMPLEX_FLOAT_TYPE>(acos(test));
    check_result<BOOST_CSTDFLOAT_EXTENDED_COMPLEX_FLOAT_TYPE>(atan(test));
    check_result<BOOST_CSTDFLOAT_EXTENDED_COMPLEX_FLOAT_TYPE>(exp(test));
    check_result<BOOST_CSTDFLOAT_EXTENDED_COMPLEX_FLOAT_TYPE>(log(test));
    check_result<BOOST_CSTDFLOAT_EXTENDED_COMPLEX_FLOAT_TYPE>(log10(test));
    check_result<BOOST_CSTDFLOAT_EXTENDED_COMPLEX_FLOAT_TYPE>(pow(test, 0));
    check_result<BOOST_CSTDFLOAT_EXTENDED_COMPLEX_FLOAT_TYPE>(sinh(test));
    check_result<BOOST_CSTDFLOAT_EXTENDED_COMPLEX_FLOAT_TYPE>(cosh(test));
    check_result<BOOST_CSTDFLOAT_EXTENDED_COMPLEX_FLOAT_TYPE>(tanh(test));
    check_result<BOOST_CSTDFLOAT_EXTENDED_COMPLEX_FLOAT_TYPE>(asinh(test));
    check_result<BOOST_CSTDFLOAT_EXTENDED_COMPLEX_FLOAT_TYPE>(acosh(test));
    check_result<BOOST_CSTDFLOAT_EXTENDED_COMPLEX_FLOAT_TYPE>(atanh(test));

    #endif // BOOST_CSTDFLOAT_EXTENDED_COMPLEX_FLOAT_TYPE
}
