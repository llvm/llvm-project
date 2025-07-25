// Copyright John Maddock 2006, 2007.
// Copyright Paul A. Bristow 2007.
// Copyright Matt Borland 2024.

//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_STATS_CAUCHY_HPP
#define BOOST_STATS_CAUCHY_HPP

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4127) // conditional expression is constant
#endif

#include <boost/math/tools/config.hpp>
#include <boost/math/tools/tuple.hpp>
#include <boost/math/tools/numeric_limits.hpp>
#include <boost/math/tools/precision.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/distributions/complement.hpp>
#include <boost/math/distributions/detail/common_error_handling.hpp>
#include <boost/math/policies/policy.hpp>
#include <boost/math/policies/error_handling.hpp>

#ifndef BOOST_MATH_HAS_NVRTC
#include <boost/math/distributions/fwd.hpp>
#include <utility>
#include <cmath>
#endif

namespace boost{ namespace math
{

template <class RealType, class Policy>
class cauchy_distribution;

namespace detail
{

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED RealType cdf_imp(const cauchy_distribution<RealType, Policy>& dist, const RealType& x, bool complement)
{
   //
   // This calculates the cdf of the Cauchy distribution and/or its complement.
   //
   // This implementation uses the formula
   //
   //     cdf = atan2(1, -x)/pi
   //
   // where x is the standardized (i.e. shifted and scaled) domain variable.
   //
   BOOST_MATH_STD_USING // for ADL of std functions
   constexpr auto function = "boost::math::cdf(cauchy<%1%>&, %1%)";
   RealType result = 0;
   RealType location = dist.location();
   RealType scale = dist.scale();
   if(false == detail::check_location(function, location, &result, Policy()))
   {
     return result;
   }
   if(false == detail::check_scale(function, scale, &result, Policy()))
   {
      return result;
   }
   #ifdef BOOST_MATH_HAS_GPU_SUPPORT
   if(x > tools::max_value<RealType>())
   {
      return static_cast<RealType>((complement) ? 0 : 1);
   }
   if(x < -tools::max_value<RealType>())
   {
      return static_cast<RealType>((complement) ? 1 : 0);
   }
   #else
   if(boost::math::numeric_limits<RealType>::has_infinity && x == boost::math::numeric_limits<RealType>::infinity())
   { // cdf +infinity is unity.
     return static_cast<RealType>((complement) ? 0 : 1);
   }
   if(boost::math::numeric_limits<RealType>::has_infinity && x == -boost::math::numeric_limits<RealType>::infinity())
   { // cdf -infinity is zero.
     return static_cast<RealType>((complement) ? 1 : 0);
   }
   #endif
   if(false == detail::check_x(function, x, &result, Policy()))
   { // Catches x == NaN
      return result;
   }
   RealType x_std = static_cast<RealType>((complement) ? 1 : -1)*(x - location) / scale;
   return atan2(static_cast<RealType>(1), x_std) / constants::pi<RealType>();
} // cdf

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED RealType quantile_imp(
      const cauchy_distribution<RealType, Policy>& dist,
      RealType p,
      bool complement)
{
   // This routine implements the quantile for the Cauchy distribution,
   // the value p may be the probability, or its complement if complement=true.
   //
   // The procedure calculates the distance from the
   // mid-point of the distribution.  This is either added or subtracted
   // from the location parameter depending on whether `complement` is true.
   //
   constexpr auto function = "boost::math::quantile(cauchy<%1%>&, %1%)";
   BOOST_MATH_STD_USING // for ADL of std functions

   RealType result = 0;
   RealType location = dist.location();
   RealType scale = dist.scale();
   if(false == detail::check_location(function, location, &result, Policy()))
   {
     return result;
   }
   if(false == detail::check_scale(function, scale, &result, Policy()))
   {
      return result;
   }
   if(false == detail::check_probability(function, p, &result, Policy()))
   {
      return result;
   }
   // Special cases:
   if(p == 1)
   {
      return (complement ? -1 : 1) * policies::raise_overflow_error<RealType>(function, 0, Policy());
   }
   if(p == 0)
   {
      return (complement ? 1 : -1) * policies::raise_overflow_error<RealType>(function, 0, Policy());
   }

   if(p > 0.5)
   {
      p = p - 1;
   }
   if(p == 0.5)   // special case:
   {
      return location;
   }
   result = -scale / tan(constants::pi<RealType>() * p);
   return complement ? RealType(location - result) : RealType(location + result);
} // quantile

} // namespace detail

template <class RealType = double, class Policy = policies::policy<> >
class cauchy_distribution
{
public:
   typedef RealType value_type;
   typedef Policy policy_type;

   BOOST_MATH_GPU_ENABLED cauchy_distribution(RealType l_location = 0, RealType l_scale = 1)
      : m_a(l_location), m_hg(l_scale)
   {
    constexpr auto function = "boost::math::cauchy_distribution<%1%>::cauchy_distribution";
     RealType result;
     detail::check_location(function, l_location, &result, Policy());
     detail::check_scale(function, l_scale, &result, Policy());
   } // cauchy_distribution

   BOOST_MATH_GPU_ENABLED RealType location()const
   {
      return m_a;
   }
   BOOST_MATH_GPU_ENABLED RealType scale()const
   {
      return m_hg;
   }

private:
   RealType m_a;    // The location, this is the median of the distribution.
   RealType m_hg;   // The scale )or shape), this is the half width at half height.
};

typedef cauchy_distribution<double> cauchy;

#ifdef __cpp_deduction_guides
template <class RealType>
cauchy_distribution(RealType)->cauchy_distribution<typename boost::math::tools::promote_args<RealType>::type>;
template <class RealType>
cauchy_distribution(RealType,RealType)->cauchy_distribution<typename boost::math::tools::promote_args<RealType>::type>;
#endif

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline const boost::math::pair<RealType, RealType> range(const cauchy_distribution<RealType, Policy>&)
{ // Range of permissible values for random variable x.
  BOOST_MATH_IF_CONSTEXPR (boost::math::numeric_limits<RealType>::has_infinity)
  { 
     return boost::math::pair<RealType, RealType>(-boost::math::numeric_limits<RealType>::infinity(), boost::math::numeric_limits<RealType>::infinity()); // - to + infinity.
  }
  else
  { // Can only use max_value.
   using boost::math::tools::max_value;
   return boost::math::pair<RealType, RealType>(-max_value<RealType>(), max_value<RealType>()); // - to + max.
  }
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline const boost::math::pair<RealType, RealType> support(const cauchy_distribution<RealType, Policy>& )
{ // Range of supported values for random variable x.
   // This is range where cdf rises from 0 to 1, and outside it, the pdf is zero.
  BOOST_MATH_IF_CONSTEXPR (boost::math::numeric_limits<RealType>::has_infinity)
  { 
     return boost::math::pair<RealType, RealType>(-boost::math::numeric_limits<RealType>::infinity(), boost::math::numeric_limits<RealType>::infinity()); // - to + infinity.
  }
  else
  { // Can only use max_value.
     using boost::math::tools::max_value;
     return boost::math::pair<RealType, RealType>(-tools::max_value<RealType>(), max_value<RealType>()); // - to + max.
  }
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType pdf(const cauchy_distribution<RealType, Policy>& dist, const RealType& x)
{  
   BOOST_MATH_STD_USING  // for ADL of std functions

   constexpr auto function = "boost::math::pdf(cauchy<%1%>&, %1%)";
   RealType result = 0;
   RealType location = dist.location();
   RealType scale = dist.scale();
   if(false == detail::check_scale(function, scale, &result, Policy()))
   {
      return result;
   }
   if(false == detail::check_location(function, location, &result, Policy()))
   {
      return result;
   }
   if((boost::math::isinf)(x))
   {
     return 0; // pdf + and - infinity is zero.
   }
   // These produce MSVC 4127 warnings, so the above used instead.
   //if(boost::math::numeric_limits<RealType>::has_infinity && abs(x) == boost::math::numeric_limits<RealType>::infinity())
   //{ // pdf + and - infinity is zero.
   //  return 0;
   //}

   if(false == detail::check_x(function, x, &result, Policy()))
   { // Catches x = NaN
      return result;
   }

   RealType xs = (x - location) / scale;
   result = 1 / (constants::pi<RealType>() * scale * (1 + xs * xs));
   return result;
} // pdf

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType cdf(const cauchy_distribution<RealType, Policy>& dist, const RealType& x)
{
   return detail::cdf_imp(dist, x, false);
} // cdf

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType quantile(const cauchy_distribution<RealType, Policy>& dist, const RealType& p)
{
   return detail::quantile_imp(dist, p, false);
} // quantile

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType cdf(const complemented2_type<cauchy_distribution<RealType, Policy>, RealType>& c)
{
   return detail::cdf_imp(c.dist, c.param, true);
} //  cdf complement

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType quantile(const complemented2_type<cauchy_distribution<RealType, Policy>, RealType>& c)
{
   return detail::quantile_imp(c.dist, c.param, true);
} // quantile complement

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType mean(const cauchy_distribution<RealType, Policy>&)
{  // There is no mean:
   typedef typename Policy::assert_undefined_type assert_type;
   static_assert(assert_type::value == 0, "The Cauchy Distribution has no mean");

   return policies::raise_domain_error<RealType>(
      "boost::math::mean(cauchy<%1%>&)",
      "The Cauchy distribution does not have a mean: "
      "the only possible return value is %1%.",
      boost::math::numeric_limits<RealType>::quiet_NaN(), Policy());
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType variance(const cauchy_distribution<RealType, Policy>& /*dist*/)
{
   // There is no variance:
   typedef typename Policy::assert_undefined_type assert_type;
   static_assert(assert_type::value == 0, "The Cauchy Distribution has no variance");

   return policies::raise_domain_error<RealType>(
      "boost::math::variance(cauchy<%1%>&)",
      "The Cauchy distribution does not have a variance: "
      "the only possible return value is %1%.",
      boost::math::numeric_limits<RealType>::quiet_NaN(), Policy());
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType mode(const cauchy_distribution<RealType, Policy>& dist)
{
   return dist.location();
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType median(const cauchy_distribution<RealType, Policy>& dist)
{
   return dist.location();
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType skewness(const cauchy_distribution<RealType, Policy>& /*dist*/)
{
   // There is no skewness:
   typedef typename Policy::assert_undefined_type assert_type;
   static_assert(assert_type::value == 0, "The Cauchy Distribution has no skewness");

   return policies::raise_domain_error<RealType>(
      "boost::math::skewness(cauchy<%1%>&)",
      "The Cauchy distribution does not have a skewness: "
      "the only possible return value is %1%.",
      boost::math::numeric_limits<RealType>::quiet_NaN(), Policy()); // infinity?
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType kurtosis(const cauchy_distribution<RealType, Policy>& /*dist*/)
{
   // There is no kurtosis:
   typedef typename Policy::assert_undefined_type assert_type;
   static_assert(assert_type::value == 0, "The Cauchy Distribution has no kurtosis");

   return policies::raise_domain_error<RealType>(
      "boost::math::kurtosis(cauchy<%1%>&)",
      "The Cauchy distribution does not have a kurtosis: "
      "the only possible return value is %1%.",
      boost::math::numeric_limits<RealType>::quiet_NaN(), Policy());
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType kurtosis_excess(const cauchy_distribution<RealType, Policy>& /*dist*/)
{
   // There is no kurtosis excess:
   typedef typename Policy::assert_undefined_type assert_type;
   static_assert(assert_type::value == 0, "The Cauchy Distribution has no kurtosis excess");

   return policies::raise_domain_error<RealType>(
      "boost::math::kurtosis_excess(cauchy<%1%>&)",
      "The Cauchy distribution does not have a kurtosis: "
      "the only possible return value is %1%.",
      boost::math::numeric_limits<RealType>::quiet_NaN(), Policy());
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType entropy(const cauchy_distribution<RealType, Policy> & dist)
{
   using std::log;
   return log(2*constants::two_pi<RealType>()*dist.scale());
}

} // namespace math
} // namespace boost

#ifdef _MSC_VER
#pragma warning(pop)
#endif

// This include must be at the end, *after* the accessors
// for this distribution have been defined, in order to
// keep compilers that support two-phase lookup happy.
#include <boost/math/distributions/detail/derived_accessors.hpp>

#endif // BOOST_STATS_CAUCHY_HPP
