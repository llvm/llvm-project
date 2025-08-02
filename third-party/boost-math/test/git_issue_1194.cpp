//  (C) Copyright Matt Borland 2024.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "math_unit_test.hpp"
#include <boost/math/special_functions/gamma.hpp>
#include <cerrno>

int main()
{
    using c99_error_policy = ::boost::math::policies::policy<
            ::boost::math::policies::domain_error< ::boost::math::policies::errno_on_error>,
            ::boost::math::policies::pole_error< ::boost::math::policies::errno_on_error>,
            ::boost::math::policies::overflow_error< ::boost::math::policies::errno_on_error>,
            ::boost::math::policies::evaluation_error< ::boost::math::policies::errno_on_error>,
            ::boost::math::policies::rounding_error< ::boost::math::policies::errno_on_error> >;

    double val = -std::numeric_limits<double>::infinity();

    val = boost::math::tgamma(val, c99_error_policy());
    CHECK_EQUAL(errno, EDOM);

    val = std::numeric_limits<double>::quiet_NaN();
    val = boost::math::tgamma(val, c99_error_policy());
    CHECK_EQUAL(errno, EDOM);

    val = std::numeric_limits<double>::infinity();
    val = boost::math::tgamma(val, c99_error_policy());
    CHECK_EQUAL(errno, ERANGE);

    val = 0;
    val = boost::math::tgamma(val, c99_error_policy());
    CHECK_EQUAL(errno, EDOM); // OK

    val = -2;
    val = boost::math::tgamma(val, c99_error_policy());
    CHECK_EQUAL(errno, EDOM); // OK

    return boost::math::test::report_errors();
}
