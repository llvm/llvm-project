// Copyright Matt Borland 2025.
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_SF_LOGIT_HPP
#define BOOST_MATH_SF_LOGIT_HPP

#include <boost/math/tools/config.hpp>
#include <boost/math/policies/policy.hpp>
#include <boost/math/policies/error_handling.hpp>
#include <cmath>
#include <cfenv>

namespace boost {
namespace math {

template <typename RealType, typename Policy>
BOOST_MATH_GPU_ENABLED RealType logit(RealType p, const Policy&)
{
    BOOST_MATH_STD_USING
    using std::atanh;

    using promoted_real_type = typename policies::evaluation<RealType, Policy>::type;

    if (p < tools::min_value<RealType>())
    {
        return -policies::raise_overflow_error<RealType>("logit", "sub-normals will overflow ln(x/(1-x))", Policy());
    }

    static const RealType crossover {RealType{1}/4};
    const auto promoted_p {static_cast<promoted_real_type>(p)};
    RealType result {};
    if (p > crossover)
    {
        result = static_cast<RealType>(2 * atanh(2 * promoted_p - 1));
    }
    else
    {
        result = static_cast<RealType>(log(promoted_p / (1 - promoted_p)));
    }

    return result;
}

template <typename RealType>
BOOST_MATH_GPU_ENABLED RealType logit(RealType p)
{
    return logit(p, policies::policy<>());
}

} // namespace math
} // namespace boost

#endif // BOOST_MATH_SF_LOGIT_HPP
