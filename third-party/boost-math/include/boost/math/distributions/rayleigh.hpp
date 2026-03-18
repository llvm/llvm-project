//  Copyright Paul A. Bristow 2007.
//  Copyright Matt Borland 2023.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_STATS_rayleigh_HPP
#define BOOST_STATS_rayleigh_HPP

#include <boost/math/tools/config.hpp>
#include <boost/math/tools/cstdint.hpp>
#include <boost/math/tools/tuple.hpp>
#include <boost/math/tools/numeric_limits.hpp>
#include <boost/math/distributions/fwd.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/log1p.hpp>
#include <boost/math/special_functions/expm1.hpp>
#include <boost/math/distributions/complement.hpp>
#include <boost/math/distributions/detail/common_error_handling.hpp>

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable: 4702) // unreachable code (return after domain_error throw).
#endif

namespace boost{ namespace math{

namespace detail
{ // Error checks:
  template <class RealType, class Policy>
  BOOST_MATH_GPU_ENABLED inline bool verify_sigma(const char* function, RealType sigma, RealType* presult, const Policy& pol)
  {
     if((sigma <= 0) || (!(boost::math::isfinite)(sigma)))
     {
        *presult = policies::raise_domain_error<RealType>(
           function,
           "The scale parameter \"sigma\" must be > 0 and finite, but was: %1%.", sigma, pol);
        return false;
     }
     return true;
  } // bool verify_sigma

  template <class RealType, class Policy>
  BOOST_MATH_GPU_ENABLED inline bool verify_rayleigh_x(const char* function, RealType x, RealType* presult, const Policy& pol)
  {
     if((x < 0) || (boost::math::isnan)(x))
     {
        *presult = policies::raise_domain_error<RealType>(
           function,
           "The random variable must be >= 0, but was: %1%.", x, pol);
        return false;
     }
     return true;
  } // bool verify_rayleigh_x
} // namespace detail

template <class RealType = double, class Policy = policies::policy<> >
class rayleigh_distribution
{
public:
   using value_type = RealType;
   using policy_type = Policy;

   BOOST_MATH_GPU_ENABLED explicit rayleigh_distribution(RealType l_sigma = 1)
      : m_sigma(l_sigma)
   {
      RealType err;
      detail::verify_sigma("boost::math::rayleigh_distribution<%1%>::rayleigh_distribution", l_sigma, &err, Policy());
   } // rayleigh_distribution

   BOOST_MATH_GPU_ENABLED RealType sigma()const
   { // Accessor.
     return m_sigma;
   }

private:
   RealType m_sigma;
}; // class rayleigh_distribution

using rayleigh = rayleigh_distribution<double>;

#ifdef __cpp_deduction_guides
template <class RealType>
rayleigh_distribution(RealType)->rayleigh_distribution<typename boost::math::tools::promote_args<RealType>::type>;
#endif

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline boost::math::pair<RealType, RealType> range(const rayleigh_distribution<RealType, Policy>& /*dist*/)
{ // Range of permissible values for random variable x.
   using boost::math::tools::max_value;
   return boost::math::pair<RealType, RealType>(static_cast<RealType>(0), boost::math::numeric_limits<RealType>::has_infinity ? boost::math::numeric_limits<RealType>::infinity() : max_value<RealType>());
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline boost::math::pair<RealType, RealType> support(const rayleigh_distribution<RealType, Policy>& /*dist*/)
{ // Range of supported values for random variable x.
   // This is range where cdf rises from 0 to 1, and outside it, the pdf is zero.
   using boost::math::tools::max_value;
   return boost::math::pair<RealType, RealType>(static_cast<RealType>(0),  max_value<RealType>());
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType pdf(const rayleigh_distribution<RealType, Policy>& dist, const RealType& x)
{
   BOOST_MATH_STD_USING // for ADL of std function exp.

   RealType sigma = dist.sigma();
   RealType result = 0;
   constexpr auto function = "boost::math::pdf(const rayleigh_distribution<%1%>&, %1%)";
   if(false == detail::verify_sigma(function, sigma, &result, Policy()))
   {
      return result;
   }
   if(false == detail::verify_rayleigh_x(function, x, &result, Policy()))
   {
      return result;
   }
   if((boost::math::isinf)(x))
   {
      return 0;
   }
   RealType sigmasqr = sigma * sigma;
   result = x * (exp(-(x * x) / ( 2 * sigmasqr))) / sigmasqr;
   return result;
} // pdf

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType logpdf(const rayleigh_distribution<RealType, Policy>& dist, const RealType& x)
{
   BOOST_MATH_STD_USING // for ADL of std function exp.

   const RealType sigma = dist.sigma();
   RealType result = -boost::math::numeric_limits<RealType>::infinity();
   constexpr auto function = "boost::math::logpdf(const rayleigh_distribution<%1%>&, %1%)";

   if(false == detail::verify_sigma(function, sigma, &result, Policy()))
   {
      return result;
   }
   if(false == detail::verify_rayleigh_x(function, x, &result, Policy()))
   {
      return result;
   }
   if((boost::math::isinf)(x))
   {
      return result;
   }

   result = -(x*x)/(2*sigma*sigma) - 2*log(sigma) + log(x);
   return result;
} // logpdf

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType cdf(const rayleigh_distribution<RealType, Policy>& dist, const RealType& x)
{
   BOOST_MATH_STD_USING // for ADL of std functions

   RealType result = 0;
   RealType sigma = dist.sigma();
   constexpr auto function = "boost::math::cdf(const rayleigh_distribution<%1%>&, %1%)";
   if(false == detail::verify_sigma(function, sigma, &result, Policy()))
   {
      return result;
   }
   if(false == detail::verify_rayleigh_x(function, x, &result, Policy()))
   {
      return result;
   }
   result = -boost::math::expm1(-x * x / ( 2 * sigma * sigma), Policy());
   return result;
} // cdf

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType logcdf(const rayleigh_distribution<RealType, Policy>& dist, const RealType& x)
{
   BOOST_MATH_STD_USING // for ADL of std functions

   RealType result = 0;
   RealType sigma = dist.sigma();
   constexpr auto function = "boost::math::logcdf(const rayleigh_distribution<%1%>&, %1%)";
   if(false == detail::verify_sigma(function, sigma, &result, Policy()))
   {
      return -boost::math::numeric_limits<RealType>::infinity();
   }
   if(false == detail::verify_rayleigh_x(function, x, &result, Policy()))
   {
      return -boost::math::numeric_limits<RealType>::infinity();
   }
   result = log1p(-exp(-x * x / ( 2 * sigma * sigma)), Policy());   
   return result;
} // logcdf

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType quantile(const rayleigh_distribution<RealType, Policy>& dist, const RealType& p)
{
   BOOST_MATH_STD_USING // for ADL of std functions

   RealType result = 0;
   RealType sigma = dist.sigma();
   constexpr auto function = "boost::math::quantile(const rayleigh_distribution<%1%>&, %1%)";
   if(false == detail::verify_sigma(function, sigma, &result, Policy()))
      return result;
   if(false == detail::check_probability(function, p, &result, Policy()))
      return result;

   if(p == 0)
   {
      return 0;
   }
   if(p == 1)
   {
     return policies::raise_overflow_error<RealType>(function, 0, Policy());
   }
   result = sqrt(-2 * sigma * sigma * boost::math::log1p(-p, Policy()));
   return result;
} // quantile

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType cdf(const complemented2_type<rayleigh_distribution<RealType, Policy>, RealType>& c)
{
   BOOST_MATH_STD_USING // for ADL of std functions

   RealType result = 0;
   RealType sigma = c.dist.sigma();
   constexpr auto function = "boost::math::cdf(const rayleigh_distribution<%1%>&, %1%)";
   if(false == detail::verify_sigma(function, sigma, &result, Policy()))
   {
      return result;
   }
   RealType x = c.param;
   if(false == detail::verify_rayleigh_x(function, x, &result, Policy()))
   {
      return result;
   }
   RealType ea = x * x / (2 * sigma * sigma);
   // Fix for VC11/12 x64 bug in exp(float):
   if (ea >= tools::max_value<RealType>())
      return 0;
   result =  exp(-ea);
   return result;
} // cdf complement

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType logcdf(const complemented2_type<rayleigh_distribution<RealType, Policy>, RealType>& c)
{
   BOOST_MATH_STD_USING // for ADL of std functions

   RealType result = 0;
   RealType sigma = c.dist.sigma();
   constexpr auto function = "boost::math::logcdf(const rayleigh_distribution<%1%>&, %1%)";
   if(false == detail::verify_sigma(function, sigma, &result, Policy()))
   {
      return -boost::math::numeric_limits<RealType>::infinity();
   }
   RealType x = c.param;
   if(false == detail::verify_rayleigh_x(function, x, &result, Policy()))
   {
      return -boost::math::numeric_limits<RealType>::infinity();
   }
   RealType ea = x * x / (2 * sigma * sigma);
   // Fix for VC11/12 x64 bug in exp(float):
   if (ea >= tools::max_value<RealType>())
      return 0;
   result = -ea;
   return result;
} // logcdf complement

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType quantile(const complemented2_type<rayleigh_distribution<RealType, Policy>, RealType>& c)
{
   BOOST_MATH_STD_USING // for ADL of std functions, log & sqrt.

   RealType result = 0;
   RealType sigma = c.dist.sigma();
   constexpr auto function = "boost::math::quantile(const rayleigh_distribution<%1%>&, %1%)";
   if(false == detail::verify_sigma(function, sigma, &result, Policy()))
   {
      return result;
   }
   RealType q = c.param;
   if(false == detail::check_probability(function, q, &result, Policy()))
   {
      return result;
   }
   if(q == 1)
   {
      return 0;
   }
   if(q == 0)
   {
     return policies::raise_overflow_error<RealType>(function, 0, Policy());
   }
   result = sqrt(-2 * sigma * sigma * log(q));
   return result;
} // quantile complement

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType mean(const rayleigh_distribution<RealType, Policy>& dist)
{
   RealType result = 0;
   RealType sigma = dist.sigma();
   constexpr auto function = "boost::math::mean(const rayleigh_distribution<%1%>&, %1%)";
   if(false == detail::verify_sigma(function, sigma, &result, Policy()))
   {
      return result;
   }
   using boost::math::constants::root_half_pi;
   return sigma * root_half_pi<RealType>();
} // mean

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType variance(const rayleigh_distribution<RealType, Policy>& dist)
{
   RealType result = 0;
   RealType sigma = dist.sigma();
   constexpr auto function = "boost::math::variance(const rayleigh_distribution<%1%>&, %1%)";
   if(false == detail::verify_sigma(function, sigma, &result, Policy()))
   {
      return result;
   }
   using boost::math::constants::four_minus_pi;
   return four_minus_pi<RealType>() * sigma * sigma / 2;
} // variance

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType mode(const rayleigh_distribution<RealType, Policy>& dist)
{
   return dist.sigma();
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType median(const rayleigh_distribution<RealType, Policy>& dist)
{
   using boost::math::constants::root_ln_four;
   return root_ln_four<RealType>() * dist.sigma();
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType skewness(const rayleigh_distribution<RealType, Policy>& /*dist*/)
{
  return static_cast<RealType>(0.63111065781893713819189935154422777984404221106391L);
  // Computed using NTL at 150 bit, about 50 decimal digits.
  // 2 * sqrt(pi) * (pi-3) / pow(4, 2/3) - pi
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType kurtosis(const rayleigh_distribution<RealType, Policy>& /*dist*/)
{
  return static_cast<RealType>(3.2450893006876380628486604106197544154170667057995L);
  // Computed using NTL at 150 bit, about 50 decimal digits.
  // 3 - (6*pi*pi - 24*pi + 16) / pow(4-pi, 2)
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType kurtosis_excess(const rayleigh_distribution<RealType, Policy>& /*dist*/)
{
  return static_cast<RealType>(0.2450893006876380628486604106197544154170667057995L);
  // Computed using NTL at 150 bit, about 50 decimal digits.
  // -(6*pi*pi - 24*pi + 16) / pow(4-pi,2)
} // kurtosis_excess

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType entropy(const rayleigh_distribution<RealType, Policy>& dist)
{
   BOOST_MATH_STD_USING
   return 1 + log(dist.sigma()*constants::one_div_root_two<RealType>()) + constants::euler<RealType>()/2;
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

#endif // BOOST_STATS_rayleigh_HPP
