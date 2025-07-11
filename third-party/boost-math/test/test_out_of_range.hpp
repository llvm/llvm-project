// Copyright John Maddock 2012.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_TEST_OUT_OF_RANGE_HPP
#define BOOST_MATH_TEST_OUT_OF_RANGE_HPP

#include <boost/math/tools/config.hpp>
#include <boost/math/special_functions/next.hpp>
#include <boost/test/unit_test.hpp>

/*` check_out_of_range functions check that bad parameters
passed to constructors and functions throw domain_error exceptions.

Usage is `check_out_of_range<DistributionType >(list-of-params);`
Where list-of-params is a list of *valid* parameters from which the distribution can be constructed
- ie the same number of args are passed to the function,
as are passed to the distribution constructor.

Checks:

* Infinity or NaN passed in place of each of the valid params.
* Infinity or NaN as a random variable.
* Out-of-range random variable passed to pdf and cdf (ie outside of "range(distro)").
* Out-of-range probability passed to quantile function and complement.

but does *not* check finite but out-of-range parameters to the constructor
because these are specific to each distribution.
*/

#if defined(BOOST_CHECK_THROW) && defined(BOOST_MATH_NO_EXCEPTIONS)
#  undef BOOST_CHECK_THROW
#  define BOOST_CHECK_THROW(x, y)
#endif

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4127)
#endif

//! \tparam Distro distribution class name, for example: @c students_t_distribution<RealType>.
//! \tparam Infinite only true if support includes infinity (default false means do not allow infinity).
template <class Distro>
void check_support(const Distro& d, bool Infinite = false)
{ // Checks that support and function calls are within expected limits.
   typedef typename Distro::value_type value_type;
   if (Infinite == false)
   {
     if ((boost::math::isfinite)(range(d).first) && (range(d).first != -boost::math::tools::max_value<value_type>()))
     { // If possible, check that a random variable value just less than the bottom of the supported range throws domain errors.
       value_type m = (range(d).first == 0) ? -boost::math::tools::min_value<value_type>() : boost::math::float_prior(range(d).first);
       BOOST_MATH_ASSERT(m != range(d).first);
       BOOST_MATH_ASSERT(m < range(d).first);
       BOOST_CHECK_THROW(pdf(d, m), std::domain_error);
       BOOST_CHECK_THROW(cdf(d, m), std::domain_error);
       BOOST_CHECK_THROW(cdf(complement(d, m)), std::domain_error);
     }
     if ((boost::math::isfinite)(range(d).second) && (range(d).second != boost::math::tools::max_value<value_type>()))
     { // If possible, check that a random variable value just more than the top of the supported range throws domain errors.
       value_type m = (range(d).second == 0) ? boost::math::tools::min_value<value_type>() : boost::math::float_next(range(d).second);
       BOOST_MATH_ASSERT(m != range(d).first);
       BOOST_MATH_ASSERT(m > range(d).first);
       BOOST_CHECK_THROW(pdf(d, m), std::domain_error);
       BOOST_CHECK_THROW(cdf(d, m), std::domain_error);
       BOOST_CHECK_THROW(cdf(complement(d, m)), std::domain_error);
     }
     if (std::numeric_limits<value_type>::has_infinity)
     { // Infinity is available,
       if ((boost::math::isfinite)(range(d).second))
       {  // and top of range doesn't include infinity,
          // check that using infinity throws domain errors.
         BOOST_CHECK_THROW(pdf(d, std::numeric_limits<value_type>::infinity()), std::domain_error);
         BOOST_CHECK_THROW(cdf(d, std::numeric_limits<value_type>::infinity()), std::domain_error);
         BOOST_CHECK_THROW(cdf(complement(d, std::numeric_limits<value_type>::infinity())), std::domain_error);
       }
       if ((boost::math::isfinite)(range(d).first))
       {  // and bottom of range doesn't include infinity,
          // check that using infinity throws domain_error exception.
         BOOST_CHECK_THROW(pdf(d, -std::numeric_limits<value_type>::infinity()), std::domain_error);
         BOOST_CHECK_THROW(cdf(d, -std::numeric_limits<value_type>::infinity()), std::domain_error);
         BOOST_CHECK_THROW(cdf(complement(d, -std::numeric_limits<value_type>::infinity())), std::domain_error);
       }
       // Check that using infinity with quantiles always throws domain_error exception.
       BOOST_CHECK_THROW(quantile(d, std::numeric_limits<value_type>::infinity()), std::domain_error);
       BOOST_CHECK_THROW(quantile(d, -std::numeric_limits<value_type>::infinity()), std::domain_error);
       BOOST_CHECK_THROW(quantile(complement(d, std::numeric_limits<value_type>::infinity())), std::domain_error);
       BOOST_CHECK_THROW(quantile(complement(d, -std::numeric_limits<value_type>::infinity())), std::domain_error);
     }
   }
   if(std::numeric_limits<value_type>::has_quiet_NaN)
   { // NaN is available.
      BOOST_CHECK_THROW(pdf(d, std::numeric_limits<value_type>::quiet_NaN()), std::domain_error);
      BOOST_CHECK_THROW(cdf(d, std::numeric_limits<value_type>::quiet_NaN()), std::domain_error);
      BOOST_CHECK_THROW(cdf(complement(d, std::numeric_limits<value_type>::quiet_NaN())), std::domain_error);
      BOOST_CHECK_THROW(pdf(d, -std::numeric_limits<value_type>::quiet_NaN()), std::domain_error);
      BOOST_CHECK_THROW(cdf(d, -std::numeric_limits<value_type>::quiet_NaN()), std::domain_error);
      BOOST_CHECK_THROW(cdf(complement(d, -std::numeric_limits<value_type>::quiet_NaN())), std::domain_error);
      BOOST_CHECK_THROW(quantile(d, std::numeric_limits<value_type>::quiet_NaN()), std::domain_error);
      BOOST_CHECK_THROW(quantile(d, -std::numeric_limits<value_type>::quiet_NaN()), std::domain_error);
      BOOST_CHECK_THROW(quantile(complement(d, std::numeric_limits<value_type>::quiet_NaN())), std::domain_error);
      BOOST_CHECK_THROW(quantile(complement(d, -std::numeric_limits<value_type>::quiet_NaN())), std::domain_error);
   }
   // Check that using probability outside [0,1] with quantiles always throws domain_error exception.
   BOOST_CHECK_THROW(quantile(d, -1), std::domain_error);
   BOOST_CHECK_THROW(quantile(d, 2), std::domain_error);
   BOOST_CHECK_THROW(quantile(complement(d, -1)), std::domain_error);
   BOOST_CHECK_THROW(quantile(complement(d, 2)), std::domain_error);
}

// Four check_out_of_range versions for distributions with zero to 3 constructor parameters.

template <class Distro>
void check_out_of_range()
{
   Distro d;
   check_support(d);
}

template <class Distro>
void check_out_of_range(typename Distro::value_type p1)
{
   typedef typename Distro::value_type value_type;
   Distro d(p1);
   check_support(d);
   if(std::numeric_limits<value_type>::has_infinity)
   {
      BOOST_CHECK_THROW(pdf(Distro(std::numeric_limits<value_type>::infinity()), range(d).first), std::domain_error);
 //     BOOST_CHECK_THROW(pdf(Distro(std::numeric_limits<value_type>::infinity()), range(d).second), std::domain_error);
   }
   if(std::numeric_limits<value_type>::has_quiet_NaN)
   {
      BOOST_CHECK_THROW(pdf(Distro(std::numeric_limits<value_type>::quiet_NaN()), range(d).first), std::domain_error);
   }
}

template <class Distro>
void check_out_of_range(typename Distro::value_type p1, typename Distro::value_type p2)
{
   typedef typename Distro::value_type value_type;
   Distro d(p1, p2);
   check_support(d);
   if(std::numeric_limits<value_type>::has_infinity)
   {
      BOOST_CHECK_THROW(pdf(Distro(std::numeric_limits<value_type>::infinity(), p2), range(d).first), std::domain_error);
      BOOST_CHECK_THROW(pdf(Distro(p1, std::numeric_limits<value_type>::infinity()), range(d).first), std::domain_error);
   }
   if(std::numeric_limits<value_type>::has_quiet_NaN)
   {
      BOOST_CHECK_THROW(pdf(Distro(std::numeric_limits<value_type>::quiet_NaN(), p2), range(d).first), std::domain_error);
      BOOST_CHECK_THROW(pdf(Distro(p1, std::numeric_limits<value_type>::quiet_NaN()), range(d).first), std::domain_error);
   }
}

template <class Distro>
void check_out_of_range(typename Distro::value_type p1, typename Distro::value_type p2, typename Distro::value_type p3)
{
   typedef typename Distro::value_type value_type;
   Distro d(p1, p2, p3);
   check_support(d);
   if(std::numeric_limits<value_type>::has_infinity)
   {
      BOOST_CHECK_THROW(pdf(Distro(std::numeric_limits<value_type>::infinity(), p2, p3), range(d).first), std::domain_error);
      BOOST_CHECK_THROW(pdf(Distro(p1, std::numeric_limits<value_type>::infinity(), p3), range(d).first), std::domain_error);
      BOOST_CHECK_THROW(pdf(Distro(p1, p2, std::numeric_limits<value_type>::infinity()), range(d).first), std::domain_error);
   }
   if(std::numeric_limits<value_type>::has_quiet_NaN)
   {
      BOOST_CHECK_THROW(pdf(Distro(std::numeric_limits<value_type>::quiet_NaN(), p2, p3), range(d).first), std::domain_error);
      BOOST_CHECK_THROW(pdf(Distro(p1, std::numeric_limits<value_type>::quiet_NaN(), p3), range(d).first), std::domain_error);
      BOOST_CHECK_THROW(pdf(Distro(p1, p2, std::numeric_limits<value_type>::quiet_NaN()), range(d).first), std::domain_error);
   }
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#endif // BOOST_MATH_TEST_OUT_OF_RANGE_HPP
