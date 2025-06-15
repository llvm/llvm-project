//  Copyright Matt Borland 2024.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  See: https://github.com/boostorg/math/issues/1139

#include "math_unit_test.hpp"
#include <boost/math/tools/rational.hpp>

int main()
{
    const double a[] = {1.0, 1.0, 1.0, 1.0, 1.0};
    const double b[] = {1.0, 1.0, 1.0, 1.0, 1.0};

    double x1 = 1e80;
    double y1 = boost::math::tools::evaluate_rational(a, b, x1);

    double x2 = -1e80;
    double y2 = boost::math::tools::evaluate_rational(a, b, x2);
    
    CHECK_ULP_CLOSE(y1, y2, 1);

    return boost::math::test::report_errors();
}
