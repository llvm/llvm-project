//  Copyright John Maddock 2006.
//  Copyright Matt Borland 2024.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_STATS_EXTREME_VALUE_HPP
#define BOOST_STATS_EXTREME_VALUE_HPP

#include <boost/math/tools/config.hpp>
#include <boost/math/tools/numeric_limits.hpp>
#include <boost/math/tools/tuple.hpp>
#include <boost/math/tools/precision.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/log1p.hpp>
#include <boost/math/special_functions/expm1.hpp>
#include <boost/math/distributions/complement.hpp>
#include <boost/math/distributions/detail/common_error_handling.hpp>
#include <boost/math/policies/policy.hpp>
#include <boost/math/policies/error_handling.hpp>

//
// This is the maximum extreme value distribution, see
// http://www.itl.nist.gov/div898/handbook/eda/section3/eda366g.htm
// and http://mathworld.wolfram.com/ExtremeValueDistribution.html
// Also known as a Fisher-Tippett distribution, a log-Weibull
// distribution or a Gumbel distribution.

#ifndef BOOST_MATH_HAS_NVRTC
#include <boost/math/distributions/fwd.hpp>
#include <utility>
#include <cmath>
#endif

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable: 4702) // unreachable code (return after domain_error throw).
#endif

namespace boost{ namespace math{

namespace detail{
//
// Error check:
//
template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline bool verify_scale_b(const char* function, RealType b, RealType* presult, const Policy& pol)
{
   if((b <= 0) || !(boost::math::isfinite)(b))
   {
      *presult = policies::raise_domain_error<RealType>(
         function,
         "The scale parameter \"b\" must be finite and > 0, but was: %1%.", b, pol);
      return false;
   }
   return true;
}

} // namespace detail

template <class RealType = double, class Policy = policies::policy<> >
class extreme_value_distribution
{
public:
   using value_type = RealType;
   using policy_type = Policy;

   BOOST_MATH_GPU_ENABLED explicit extreme_value_distribution(RealType a = 0, RealType b = 1)
      : m_a(a), m_b(b)
   {
      RealType err;
      detail::verify_scale_b("boost::math::extreme_value_distribution<%1%>::extreme_value_distribution", b, &err, Policy());
      detail::check_finite("boost::math::extreme_value_distribution<%1%>::extreme_value_distribution", a, &err, Policy());
   } // extreme_value_distribution

   BOOST_MATH_GPU_ENABLED RealType location()const { return m_a; }
   BOOST_MATH_GPU_ENABLED RealType scale()const { return m_b; }

private:
   RealType m_a;
   RealType m_b;
};

using extreme_value = extreme_value_distribution<double>;

#ifdef __cpp_deduction_guides
template <class RealType>
extreme_value_distribution(RealType)->extreme_value_distribution<typename boost::math::tools::promote_args<RealType>::type>;
template <class RealType>
extreme_value_distribution(RealType,RealType)->extreme_value_distribution<typename boost::math::tools::promote_args<RealType>::type>;
#endif

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline boost::math::pair<RealType, RealType> range(const extreme_value_distribution<RealType, Policy>& /*dist*/)
{ // Range of permissible values for random variable x.
   using boost::math::tools::max_value;
   return boost::math::pair<RealType, RealType>(
      boost::math::numeric_limits<RealType>::has_infinity ? -boost::math::numeric_limits<RealType>::infinity() : -max_value<RealType>(), 
      boost::math::numeric_limits<RealType>::has_infinity ? boost::math::numeric_limits<RealType>::infinity() : max_value<RealType>());
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline boost::math::pair<RealType, RealType> support(const extreme_value_distribution<RealType, Policy>& /*dist*/)
{ // Range of supported values for random variable x.
   // This is range where cdf rises from 0 to 1, and outside it, the pdf is zero.
   using boost::math::tools::max_value;
   return boost::math::pair<RealType, RealType>(-max_value<RealType>(),  max_value<RealType>());
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType pdf(const extreme_value_distribution<RealType, Policy>& dist, const RealType& x)
{
   BOOST_MATH_STD_USING // for ADL of std functions

   constexpr auto function = "boost::math::pdf(const extreme_value_distribution<%1%>&, %1%)";

   RealType a = dist.location();
   RealType b = dist.scale();
   RealType result = 0;
   if(0 == detail::verify_scale_b(function, b, &result, Policy()))
      return result;
   if(0 == detail::check_finite(function, a, &result, Policy()))
      return result;
   if((boost::math::isinf)(x))
      return 0.0f;
   if(0 == detail::check_x(function, x, &result, Policy()))
      return result;
   RealType e = (a - x) / b;
   if(e < tools::log_max_value<RealType>())
      result = exp(e) * exp(-exp(e)) / b;
   // else.... result *must* be zero since exp(e) is infinite...
   return result;
} // pdf

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType logpdf(const extreme_value_distribution<RealType, Policy>& dist, const RealType& x)
{
   BOOST_MATH_STD_USING // for ADL of std functions

   constexpr auto function = "boost::math::logpdf(const extreme_value_distribution<%1%>&, %1%)";

   RealType a = dist.location();
   RealType b = dist.scale();
   RealType result = -boost::math::numeric_limits<RealType>::infinity();
   if(0 == detail::verify_scale_b(function, b, &result, Policy()))
      return result;
   if(0 == detail::check_finite(function, a, &result, Policy()))
      return result;
   if((boost::math::isinf)(x))
      return 0.0f;
   if(0 == detail::check_x(function, x, &result, Policy()))
      return result;
   RealType e = (a - x) / b;
   if(e < tools::log_max_value<RealType>())
      result = log(1/b) + e - exp(e);
   // else.... result *must* be zero since exp(e) is infinite...
   return result;
} // logpdf

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType cdf(const extreme_value_distribution<RealType, Policy>& dist, const RealType& x)
{
   BOOST_MATH_STD_USING // for ADL of std functions

   constexpr auto function = "boost::math::cdf(const extreme_value_distribution<%1%>&, %1%)";

   if((boost::math::isinf)(x))
      return x < 0 ? 0.0f : 1.0f;
   RealType a = dist.location();
   RealType b = dist.scale();
   RealType result = 0;
   if(0 == detail::verify_scale_b(function, b, &result, Policy()))
      return result;
   if(0 == detail::check_finite(function, a, &result, Policy()))
      return result;
   if(0 == detail::check_finite(function, a, &result, Policy()))
      return result;
   if(0 == detail::check_x("boost::math::cdf(const extreme_value_distribution<%1%>&, %1%)", x, &result, Policy()))
      return result;

   result = exp(-exp((a-x)/b));

   return result;
} // cdf

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType logcdf(const extreme_value_distribution<RealType, Policy>& dist, const RealType& x)
{
   BOOST_MATH_STD_USING // for ADL of std functions

   constexpr auto function = "boost::math::logcdf(const extreme_value_distribution<%1%>&, %1%)";

   if((boost::math::isinf)(x))
      return x < 0 ? 0.0f : 1.0f;
   RealType a = dist.location();
   RealType b = dist.scale();
   RealType result = 0;
   if(0 == detail::verify_scale_b(function, b, &result, Policy()))
      return result;
   if(0 == detail::check_finite(function, a, &result, Policy()))
      return result;
   if(0 == detail::check_finite(function, a, &result, Policy()))
      return result;
   if(0 == detail::check_x("boost::math::logcdf(const extreme_value_distribution<%1%>&, %1%)", x, &result, Policy()))
      return result;

   result = -exp((a-x)/b);

   return result;
} // logcdf

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED RealType quantile(const extreme_value_distribution<RealType, Policy>& dist, const RealType& p)
{
   BOOST_MATH_STD_USING // for ADL of std functions

   constexpr auto function = "boost::math::quantile(const extreme_value_distribution<%1%>&, %1%)";

   RealType a = dist.location();
   RealType b = dist.scale();
   RealType result = 0;
   if(0 == detail::verify_scale_b(function, b, &result, Policy()))
      return result;
   if(0 == detail::check_finite(function, a, &result, Policy()))
      return result;
   if(0 == detail::check_probability(function, p, &result, Policy()))
      return result;

   if(p == 0)
      return -policies::raise_overflow_error<RealType>(function, 0, Policy());
   if(p == 1)
      return policies::raise_overflow_error<RealType>(function, 0, Policy());

   result = a - log(-log(p)) * b;

   return result;
} // quantile

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType cdf(const complemented2_type<extreme_value_distribution<RealType, Policy>, RealType>& c)
{
   BOOST_MATH_STD_USING // for ADL of std functions

   constexpr auto function = "boost::math::cdf(const extreme_value_distribution<%1%>&, %1%)";

   if((boost::math::isinf)(c.param))
      return c.param < 0 ? 1.0f : 0.0f;
   RealType a = c.dist.location();
   RealType b = c.dist.scale();
   RealType result = 0;
   if(0 == detail::verify_scale_b(function, b, &result, Policy()))
      return result;
   if(0 == detail::check_finite(function, a, &result, Policy()))
      return result;
   if(0 == detail::check_x(function, c.param, &result, Policy()))
      return result;

   result = -boost::math::expm1(-exp((a-c.param)/b), Policy());

   return result;
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType logcdf(const complemented2_type<extreme_value_distribution<RealType, Policy>, RealType>& c)
{
   BOOST_MATH_STD_USING // for ADL of std functions

   constexpr auto function = "boost::math::logcdf(const extreme_value_distribution<%1%>&, %1%)";

   if((boost::math::isinf)(c.param))
      return c.param < 0 ? 1.0f : 0.0f;
   RealType a = c.dist.location();
   RealType b = c.dist.scale();
   RealType result = 0;
   if(0 == detail::verify_scale_b(function, b, &result, Policy()))
      return result;
   if(0 == detail::check_finite(function, a, &result, Policy()))
      return result;
   if(0 == detail::check_x(function, c.param, &result, Policy()))
      return result;

   result = log1p(-exp(-exp((a-c.param)/b)), Policy());

   return result;
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED RealType quantile(const complemented2_type<extreme_value_distribution<RealType, Policy>, RealType>& c)
{
   BOOST_MATH_STD_USING // for ADL of std functions

   constexpr auto function = "boost::math::quantile(const extreme_value_distribution<%1%>&, %1%)";

   RealType a = c.dist.location();
   RealType b = c.dist.scale();
   RealType q = c.param;
   RealType result = 0;
   if(0 == detail::verify_scale_b(function, b, &result, Policy()))
      return result;
   if(0 == detail::check_finite(function, a, &result, Policy()))
      return result;
   if(0 == detail::check_probability(function, q, &result, Policy()))
      return result;

   if(q == 0)
      return policies::raise_overflow_error<RealType>(function, 0, Policy());
   if(q == 1)
      return -policies::raise_overflow_error<RealType>(function, 0, Policy());

   result = a - log(-boost::math::log1p(-q, Policy())) * b;

   return result;
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType mean(const extreme_value_distribution<RealType, Policy>& dist)
{
   RealType a = dist.location();
   RealType b = dist.scale();
   RealType result = 0;
   if(0 == detail::verify_scale_b("boost::math::mean(const extreme_value_distribution<%1%>&)", b, &result, Policy()))
      return result;
   if (0 == detail::check_finite("boost::math::mean(const extreme_value_distribution<%1%>&)", a, &result, Policy()))
      return result;
   return a + constants::euler<RealType>() * b;
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType standard_deviation(const extreme_value_distribution<RealType, Policy>& dist)
{
   BOOST_MATH_STD_USING // for ADL of std functions.

   RealType b = dist.scale();
   RealType result = 0;
   if(0 == detail::verify_scale_b("boost::math::standard_deviation(const extreme_value_distribution<%1%>&)", b, &result, Policy()))
      return result;
   if(0 == detail::check_finite("boost::math::standard_deviation(const extreme_value_distribution<%1%>&)", dist.location(), &result, Policy()))
      return result;
   return constants::pi<RealType>() * b / sqrt(static_cast<RealType>(6));
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType mode(const extreme_value_distribution<RealType, Policy>& dist)
{
   return dist.location();
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType median(const extreme_value_distribution<RealType, Policy>& dist)
{
  using constants::ln_ln_two;
   return dist.location() - dist.scale() * ln_ln_two<RealType>();
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType skewness(const extreme_value_distribution<RealType, Policy>& /*dist*/)
{
   //
   // This is 12 * sqrt(6) * zeta(3) / pi^3:
   // See http://mathworld.wolfram.com/ExtremeValueDistribution.html
   //
   return static_cast<RealType>(1.1395470994046486574927930193898461120875997958366L);
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType kurtosis(const extreme_value_distribution<RealType, Policy>& /*dist*/)
{
   // See http://mathworld.wolfram.com/ExtremeValueDistribution.html
   return RealType(27) / 5;
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType kurtosis_excess(const extreme_value_distribution<RealType, Policy>& /*dist*/)
{
   // See http://mathworld.wolfram.com/ExtremeValueDistribution.html
   return RealType(12) / 5;
}


} // namespace math
} // namespace boost

#ifdef _MSC_VER
# pragma warning(pop)
#endif

// This include must be at the end, *after* the accessors
// for this distribution have been defined, in order to
// keep compilers that support two-phase lookup happy.
#include <boost/math/distributions/detail/derived_accessors.hpp>

#endif // BOOST_STATS_EXTREME_VALUE_HPP
