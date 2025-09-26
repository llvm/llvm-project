// Copyright John Maddock 2006.
// Copyright Paul A. Bristow 2007, 2009
// Copyright Matt Borland 2024
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_MATH_OVERFLOW_ERROR_POLICY ignore_error
#define BOOST_MATH_PROMOTE_DOUBLE_POLICY false

#include <boost/math/concepts/real_concept.hpp>
#include <boost/math/special_functions/beta.hpp>
#include "math_unit_test.hpp"

template <class T>
void test_spots(T)
{
    //
    // Basic sanity checks, tolerance is 20 epsilon expressed as a percentage:
    //
    T tolerance = 20;
    T small = boost::math::tools::epsilon<T>() / 1024;
    CHECK_ULP_CLOSE(::boost::math::beta(static_cast<T>(1), static_cast<T>(1)), static_cast<T>(1), tolerance);
    CHECK_ULP_CLOSE(::boost::math::beta(static_cast<T>(1), static_cast<T>(4)), static_cast<T>(0.25), tolerance);
    CHECK_ULP_CLOSE(::boost::math::beta(static_cast<T>(4), static_cast<T>(1)), static_cast<T>(0.25), tolerance);
    CHECK_ULP_CLOSE(::boost::math::beta(small, static_cast<T>(4)), 1/small, tolerance);
    CHECK_ULP_CLOSE(::boost::math::beta(static_cast<T>(4), small), 1/small, tolerance);
    CHECK_ULP_CLOSE(::boost::math::beta(small, static_cast<T>(4)), 1/small, tolerance);
    CHECK_ULP_CLOSE(::boost::math::beta(static_cast<T>(4), static_cast<T>(20)), static_cast<T>(0.00002823263692828910220214568040654997176736L), tolerance);
}

int main()
{
    test_spots(0.0F);
    test_spots(0.0);

    return boost::math::test::report_errors();
}
