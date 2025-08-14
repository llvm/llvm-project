//  Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Basic sanity check that header
// #includes all the files that it needs to.
//
#include <boost/math/cstdfloat/cstdfloat_limits.hpp>
//
// Note this header includes no other headers, this is
// important if this test is to be meaningful:
//
#include "test_compile_result.hpp"

void compile_and_link_test()
{
    #ifdef BOOST_FLOAT128_C
    check_result<bool>(std::numeric_limits<boost::float128_t>::is_specialized);
    check_result<boost::float128_t>((std::numeric_limits<boost::float128_t>::min)());
    check_result<boost::float128_t>((std::numeric_limits<boost::float128_t>::max)());
    check_result<boost::float128_t>(std::numeric_limits<boost::float128_t>::lowest());
    check_result<int>(std::numeric_limits<boost::float128_t>::digits);
    check_result<int>(std::numeric_limits<boost::float128_t>::digits10);
    check_result<int>(std::numeric_limits<boost::float128_t>::max_digits10);
    check_result<bool>(std::numeric_limits<boost::float128_t>::is_signed);
    check_result<bool>(std::numeric_limits<boost::float128_t>::is_integer);
    check_result<bool>(std::numeric_limits<boost::float128_t>::is_exact);
    check_result<int>(std::numeric_limits<boost::float128_t>::radix);
    check_result<boost::float128_t>(std::numeric_limits<boost::float128_t>::epsilon());
    check_result<int>(std::numeric_limits<boost::float128_t>::min_exponent);
    check_result<int>(std::numeric_limits<boost::float128_t>::min_exponent10);
    check_result<int>(std::numeric_limits<boost::float128_t>::max_exponent);
    check_result<int>(std::numeric_limits<boost::float128_t>::max_exponent10);
    check_result<bool>(std::numeric_limits<boost::float128_t>::has_infinity);
    check_result<bool>(std::numeric_limits<boost::float128_t>::has_quiet_NaN);
    check_result<bool>(std::numeric_limits<boost::float128_t>::has_signaling_NaN);
    check_result<std::float_denorm_style>(std::numeric_limits<boost::float128_t>::has_denorm);
    check_result<bool>(std::numeric_limits<boost::float128_t>::has_denorm_loss);
    check_result<boost::float128_t>(std::numeric_limits<boost::float128_t>::infinity());
    check_result<boost::float128_t>(std::numeric_limits<boost::float128_t>::quiet_NaN());
    check_result<boost::float128_t>(std::numeric_limits<boost::float128_t>::signaling_NaN());
    check_result<boost::float128_t>(std::numeric_limits<boost::float128_t>::denorm_min());
    check_result<bool>(std::numeric_limits<boost::float128_t>::is_iec559);
    check_result<bool>(std::numeric_limits<boost::float128_t>::is_bounded);
    check_result<bool>(std::numeric_limits<boost::float128_t>::is_modulo);
    check_result<bool>(std::numeric_limits<boost::float128_t>::traps);
    check_result<bool>(std::numeric_limits<boost::float128_t>::tinyness_before);
    check_result<std::float_round_style>(std::numeric_limits<boost::float128_t>::round_style);
    #endif
}
