//  Copyright John Maddock 2006, 2007.
//  Copyright Paul A. Bristow 2006, 2007.
//  Copyright Matt Borland 2024.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_STATS_NORMAL_HPP
#define BOOST_STATS_NORMAL_HPP

// http://en.wikipedia.org/wiki/Normal_distribution
// http://www.itl.nist.gov/div898/handbook/eda/section3/eda3661.htm
// Also:
// Weisstein, Eric W. "Normal Distribution."
// From MathWorld--A Wolfram Web Resource.
// http://mathworld.wolfram.com/NormalDistribution.html

#include <boost/math/tools/config.hpp>
#include <boost/math/tools/tuple.hpp>
#include <boost/math/tools/numeric_limits.hpp>
#include <boost/math/distributions/fwd.hpp>
#include <boost/math/special_functions/erf.hpp> // for erf/erfc.
#include <boost/math/distributions/complement.hpp>
#include <boost/math/distributions/detail/common_error_handling.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/policies/policy.hpp>

namespace boost{ namespace math{

template <class RealType = double, class Policy = policies::policy<> >
class normal_distribution
{
public:
   using value_type = RealType;
   using policy_type = Policy;

   BOOST_MATH_GPU_ENABLED explicit normal_distribution(RealType l_mean = 0, RealType sd = 1)
      : m_mean(l_mean), m_sd(sd)
   { // Default is a 'standard' normal distribution N01.
     constexpr auto function = "boost::math::normal_distribution<%1%>::normal_distribution";

     RealType result;
     detail::check_scale(function, sd, &result, Policy());
     detail::check_location(function, l_mean, &result, Policy());
   }

   BOOST_MATH_GPU_ENABLED RealType mean()const
   { // alias for location.
      return m_mean;
   }

   BOOST_MATH_GPU_ENABLED RealType standard_deviation()const
   { // alias for scale.
      return m_sd;
   }

   // Synonyms, provided to allow generic use of find_location and find_scale.
   BOOST_MATH_GPU_ENABLED RealType location()const
   { // location.
      return m_mean;
   }
   BOOST_MATH_GPU_ENABLED RealType scale()const
   { // scale.
      return m_sd;
   }

private:
   //
   // Data members:
   //
   RealType m_mean;  // distribution mean or location.
   RealType m_sd;    // distribution standard deviation or scale.
}; // class normal_distribution

using normal = normal_distribution<double>;

//
// Deduction guides, note we don't check the 
// value of __cpp_deduction_guides, just assume
// they work as advertised, even if this is pre-final C++17.
//
#ifdef __cpp_deduction_guides

template <class RealType>
normal_distribution(RealType, RealType)->normal_distribution<typename boost::math::tools::promote_args<RealType>::type>;
template <class RealType>
normal_distribution(RealType)->normal_distribution<typename boost::math::tools::promote_args<RealType>::type>;

#endif

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4127)
#endif

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline boost::math::pair<RealType, RealType> range(const normal_distribution<RealType, Policy>& /*dist*/)
{ // Range of permissible values for random variable x.
  BOOST_MATH_IF_CONSTEXPR (boost::math::numeric_limits<RealType>::has_infinity)
  { 
     return boost::math::pair<RealType, RealType>(-boost::math::numeric_limits<RealType>::infinity(), boost::math::numeric_limits<RealType>::infinity()); // - to + infinity.
  }
  else
  { // Can only use max_value.
    using boost::math::tools::max_value;
    return boost::math::pair<RealType, RealType>(-max_value<RealType>(), max_value<RealType>()); // - to + max value.
  }
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline boost::math::pair<RealType, RealType> support(const normal_distribution<RealType, Policy>& /*dist*/)
{ // This is range values for random variable x where cdf rises from 0 to 1, and outside it, the pdf is zero.
  BOOST_MATH_IF_CONSTEXPR (boost::math::numeric_limits<RealType>::has_infinity)
  { 
     return boost::math::pair<RealType, RealType>(-boost::math::numeric_limits<RealType>::infinity(), boost::math::numeric_limits<RealType>::infinity()); // - to + infinity.
  }
  else
  { // Can only use max_value.
   using boost::math::tools::max_value;
   return boost::math::pair<RealType, RealType>(-max_value<RealType>(),  max_value<RealType>()); // - to + max value.
  }
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType pdf(const normal_distribution<RealType, Policy>& dist, const RealType& x)
{
   BOOST_MATH_STD_USING  // for ADL of std functions

   RealType sd = dist.standard_deviation();
   RealType mean = dist.mean();

   constexpr auto function = "boost::math::pdf(const normal_distribution<%1%>&, %1%)";

   RealType result = 0;
   if(false == detail::check_scale(function, sd, &result, Policy()))
   {
      return result;
   }
   if(false == detail::check_location(function, mean, &result, Policy()))
   {
      return result;
   }
   if((boost::math::isinf)(x))
   {
     return 0; // pdf + and - infinity is zero.
   }
   if(false == detail::check_x(function, x, &result, Policy()))
   {
      return result;
   }

   RealType exponent = x - mean;
   exponent *= -exponent;
   exponent /= 2 * sd * sd;

   result = exp(exponent);
   result /= sd * sqrt(2 * constants::pi<RealType>());

   return result;
} // pdf

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType logpdf(const normal_distribution<RealType, Policy>& dist, const RealType& x)
{
   BOOST_MATH_STD_USING  // for ADL of std functions

   const RealType sd = dist.standard_deviation();
   const RealType mean = dist.mean();

   constexpr auto function = "boost::math::logpdf(const normal_distribution<%1%>&, %1%)";

   RealType result = -boost::math::numeric_limits<RealType>::infinity();
   if(false == detail::check_scale(function, sd, &result, Policy()))
   {
      return result;
   }
   if(false == detail::check_location(function, mean, &result, Policy()))
   {
      return result;
   }
   if((boost::math::isinf)(x))
   {
      return result; // pdf + and - infinity is zero so logpdf is -inf
   }
   if(false == detail::check_x(function, x, &result, Policy()))
   {
      return result;
   }

   const RealType pi = boost::math::constants::pi<RealType>();
   const RealType half = boost::math::constants::half<RealType>();

   result = -log(sd) - half*log(2*pi) - (x-mean)*(x-mean)/(2*sd*sd);

   return result;
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType cdf(const normal_distribution<RealType, Policy>& dist, const RealType& x)
{
   BOOST_MATH_STD_USING  // for ADL of std functions

   RealType sd = dist.standard_deviation();
   RealType mean = dist.mean();
   constexpr auto function = "boost::math::cdf(const normal_distribution<%1%>&, %1%)";
   RealType result = 0;
   if(false == detail::check_scale(function, sd, &result, Policy()))
   {
      return result;
   }
   if(false == detail::check_location(function, mean, &result, Policy()))
   {
      return result;
   }
   if((boost::math::isinf)(x))
   {
     if(x < 0) return 0; // -infinity
     return 1; // + infinity
   }
   if(false == detail::check_x(function, x, &result, Policy()))
   {
     return result;
   }
   RealType diff = (x - mean) / (sd * constants::root_two<RealType>());
   result = boost::math::erfc(-diff, Policy()) / 2;
   return result;
} // cdf

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType quantile(const normal_distribution<RealType, Policy>& dist, const RealType& p)
{
   BOOST_MATH_STD_USING  // for ADL of std functions

   RealType sd = dist.standard_deviation();
   RealType mean = dist.mean();
   constexpr auto function = "boost::math::quantile(const normal_distribution<%1%>&, %1%)";

   RealType result = 0;
   if(false == detail::check_scale(function, sd, &result, Policy()))
      return result;
   if(false == detail::check_location(function, mean, &result, Policy()))
      return result;
   if(false == detail::check_probability(function, p, &result, Policy()))
      return result;

   result= boost::math::erfc_inv(2 * p, Policy());
   result = -result;
   result *= sd * constants::root_two<RealType>();
   result += mean;
   return result;
} // quantile

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType cdf(const complemented2_type<normal_distribution<RealType, Policy>, RealType>& c)
{
   BOOST_MATH_STD_USING  // for ADL of std functions

   RealType sd = c.dist.standard_deviation();
   RealType mean = c.dist.mean();
   RealType x = c.param;
   constexpr auto function = "boost::math::cdf(const complement(normal_distribution<%1%>&), %1%)";

   RealType result = 0;
   if(false == detail::check_scale(function, sd, &result, Policy()))
      return result;
   if(false == detail::check_location(function, mean, &result, Policy()))
      return result;
   if((boost::math::isinf)(x))
   {
     if(x < 0) return 1; // cdf complement -infinity is unity.
     return 0; // cdf complement +infinity is zero
   }
   if(false == detail::check_x(function, x, &result, Policy()))
      return result;

   RealType diff = (x - mean) / (sd * constants::root_two<RealType>());
   result = boost::math::erfc(diff, Policy()) / 2;
   return result;
} // cdf complement

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType quantile(const complemented2_type<normal_distribution<RealType, Policy>, RealType>& c)
{
   BOOST_MATH_STD_USING  // for ADL of std functions

   RealType sd = c.dist.standard_deviation();
   RealType mean = c.dist.mean();
   constexpr auto function = "boost::math::quantile(const complement(normal_distribution<%1%>&), %1%)";
   RealType result = 0;
   if(false == detail::check_scale(function, sd, &result, Policy()))
      return result;
   if(false == detail::check_location(function, mean, &result, Policy()))
      return result;
   RealType q = c.param;
   if(false == detail::check_probability(function, q, &result, Policy()))
      return result;
   result = boost::math::erfc_inv(2 * q, Policy());
   result *= sd * constants::root_two<RealType>();
   result += mean;
   return result;
} // quantile

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType mean(const normal_distribution<RealType, Policy>& dist)
{
   return dist.mean();
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType standard_deviation(const normal_distribution<RealType, Policy>& dist)
{
   return dist.standard_deviation();
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType mode(const normal_distribution<RealType, Policy>& dist)
{
   return dist.mean();
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType median(const normal_distribution<RealType, Policy>& dist)
{
   return dist.mean();
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType skewness(const normal_distribution<RealType, Policy>& /*dist*/)
{
   return 0;
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType kurtosis(const normal_distribution<RealType, Policy>& /*dist*/)
{
   return 3;
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType kurtosis_excess(const normal_distribution<RealType, Policy>& /*dist*/)
{
   return 0;
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType entropy(const normal_distribution<RealType, Policy> & dist)
{
   BOOST_MATH_STD_USING
   RealType arg = constants::two_pi<RealType>()*constants::e<RealType>()*dist.standard_deviation()*dist.standard_deviation();
   return log(arg)/2;
}

} // namespace math
} // namespace boost

// This include must be at the end, *after* the accessors
// for this distribution have been defined, in order to
// keep compilers that support two-phase lookup happy.
#include <boost/math/distributions/detail/derived_accessors.hpp>

#endif // BOOST_STATS_NORMAL_HPP


