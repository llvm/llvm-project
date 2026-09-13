// Copyright Matt Borland 2025.
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_SF_EXPIT_HPP
#define BOOST_MATH_SF_EXPIT_HPP

#include <boost/math/policies/policy.hpp>
#include <boost/math/tools/precision.hpp>
#include <cmath>

namespace boost {
namespace math {

template <typename RealType, typename Policy>
BOOST_MATH_GPU_ENABLED RealType logistic_sigmoid(RealType x, const Policy&)
{
    BOOST_MATH_STD_USING

    using promoted_real_type = typename policies::evaluation<RealType, Policy>::type;

    if(-x >= tools::log_max_value<RealType>())
    {
        return static_cast<RealType>(0);
    }
    if(-x <= -tools::log_max_value<RealType>())
    {
        return static_cast<RealType>(1);
    }

    const auto res {static_cast<RealType>(1 / (1 + exp(static_cast<promoted_real_type>(-x))))};
    return res;
}

template <typename RealType>
BOOST_MATH_GPU_ENABLED RealType logistic_sigmoid(RealType x)
{
    return logistic_sigmoid(x, policies::policy<>());
}

} // namespace math
} // namespace boost

#endif // BOOST_MATH_SF_EXPIT_HPP
