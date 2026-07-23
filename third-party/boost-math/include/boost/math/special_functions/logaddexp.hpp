//  (C) Copyright Matt Borland 2022.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cmath>
#include <limits>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/math/constants/constants.hpp>

namespace boost { namespace math {

// Calculates log(exp(x1) + exp(x2))
template <typename Real>
Real logaddexp(Real x1, Real x2) noexcept
{
    using std::log1p;
    using std::exp;
    using std::abs;
    
    // Validate inputs first
    if (!(boost::math::isfinite)(x1))
    {
        return x1;
    }
    else if (!(boost::math::isfinite)(x2))
    {
        return x2;
    }

    const Real temp = x1 - x2;

    if (temp > 0)
    {
        return x1 + log1p(exp(-temp));
    }

    return x2 + log1p(exp(temp));
}

}} // Namespace boost::math
