//  Copyright John Maddock 2006.
//  Copyright Paul A. Bristow 2006, 2012, 2017.
//  Copyright Thomas Mang 2012.
//  Copyright Matt Borland 2024.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_STATS_STUDENTS_T_HPP
#define BOOST_STATS_STUDENTS_T_HPP

// http://en.wikipedia.org/wiki/Student%27s_t_distribution
// http://www.itl.nist.gov/div898/handbook/eda/section3/eda3664.htm

#include <boost/math/tools/config.hpp>
#include <boost/math/tools/tuple.hpp>
#include <boost/math/tools/cstdint.hpp>
#include <boost/math/tools/numeric_limits.hpp>
#include <boost/math/distributions/fwd.hpp>
#include <boost/math/special_functions/beta.hpp> // for ibeta(a, b, x).
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/relative_difference.hpp>
#include <boost/math/distributions/complement.hpp>
#include <boost/math/distributions/detail/common_error_handling.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/cauchy.hpp>
#include <boost/math/policies/policy.hpp>

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable: 4702) // unreachable code (return after domain_error throw).
#endif

namespace boost { namespace math {

template <class RealType = double, class Policy = policies::policy<> >
class students_t_distribution
{
public:
   typedef RealType value_type;
   typedef Policy policy_type;

   BOOST_MATH_GPU_ENABLED students_t_distribution(RealType df) : df_(df)
   { // Constructor.
      RealType result;
      detail::check_df_gt0_to_inf( // Checks that df > 0 or df == inf.
         "boost::math::students_t_distribution<%1%>::students_t_distribution", df_, &result, Policy());
   } // students_t_distribution

   BOOST_MATH_GPU_ENABLED RealType degrees_of_freedom()const
   {
      return df_;
   }

   // Parameter estimation:
   BOOST_MATH_GPU_ENABLED static RealType find_degrees_of_freedom(
      RealType difference_from_mean,
      RealType alpha,
      RealType beta,
      RealType sd,
      RealType hint = 0);

   BOOST_MATH_GPU_ENABLED static RealType find_degrees_of_freedom(
      RealType t,
      RealType p);

private:
   // Data member:
   RealType df_;  // degrees of freedom is a real number > 0 or +infinity.
};

typedef students_t_distribution<double> students_t; // Convenience typedef for double version.

#ifdef __cpp_deduction_guides
template <class RealType>
students_t_distribution(RealType)->students_t_distribution<typename boost::math::tools::promote_args<RealType>::type>;
#endif

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline const boost::math::pair<RealType, RealType> range(const students_t_distribution<RealType, Policy>& /*dist*/)
{ // Range of permissible values for random variable x.
  // Now including infinity.
   using boost::math::tools::max_value;
   //return boost::math::pair<RealType, RealType>(-max_value<RealType>(), max_value<RealType>());
   return boost::math::pair<RealType, RealType>(((::boost::math::numeric_limits<RealType>::is_specialized & ::boost::math::numeric_limits<RealType>::has_infinity) ? -boost::math::numeric_limits<RealType>::infinity() : -max_value<RealType>()), ((::boost::math::numeric_limits<RealType>::is_specialized & ::boost::math::numeric_limits<RealType>::has_infinity) ? +boost::math::numeric_limits<RealType>::infinity() : +max_value<RealType>()));
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline const boost::math::pair<RealType, RealType> support(const students_t_distribution<RealType, Policy>& /*dist*/)
{ // Range of supported values for random variable x.
  // Now including infinity.
   // This is range where cdf rises from 0 to 1, and outside it, the pdf is zero.
   using boost::math::tools::max_value;
   //return boost::math::pair<RealType, RealType>(-max_value<RealType>(), max_value<RealType>());
   return boost::math::pair<RealType, RealType>(((::boost::math::numeric_limits<RealType>::is_specialized & ::boost::math::numeric_limits<RealType>::has_infinity) ? -boost::math::numeric_limits<RealType>::infinity() : -max_value<RealType>()), ((::boost::math::numeric_limits<RealType>::is_specialized & ::boost::math::numeric_limits<RealType>::has_infinity) ? +boost::math::numeric_limits<RealType>::infinity() : +max_value<RealType>()));
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType pdf(const students_t_distribution<RealType, Policy>& dist, const RealType& x)
{
   BOOST_FPU_EXCEPTION_GUARD
   BOOST_MATH_STD_USING  // for ADL of std functions.

   RealType error_result;
   if(false == detail::check_x_not_NaN(
      "boost::math::pdf(const students_t_distribution<%1%>&, %1%)", x, &error_result, Policy()))
      return error_result;
   RealType df = dist.degrees_of_freedom();
   if(false == detail::check_df_gt0_to_inf( // Check that df > 0 or == +infinity.
      "boost::math::pdf(const students_t_distribution<%1%>&, %1%)", df, &error_result, Policy()))
      return error_result;

   RealType result;
   if ((boost::math::isinf)(x))
   { // - or +infinity.
     result = static_cast<RealType>(0);
     return result;
   }
   RealType limit = policies::get_epsilon<RealType, Policy>();
   // Use policies so that if policy requests lower precision, 
   // then get the normal distribution approximation earlier.
   limit = static_cast<RealType>(1) / limit; // 1/eps
   // for 64-bit double 1/eps = 4503599627370496
   if (df > limit)
   { // Special case for really big degrees_of_freedom > 1 / eps 
     // - use normal distribution which is much faster and more accurate.
     normal_distribution<RealType, Policy> n(0, 1); 
     result = pdf(n, x);
   }
   else
   { // 
     if (fabs(x) > sqrt(tools::max_value<RealType>()))
     {
        // Exact limit: the pdf underflows to zero long before x * x can overflow.
        return 0;
     }
     RealType basem1 = x * x / df;
     if(basem1 < 0.125)
     {
        result = exp(-boost::math::log1p(basem1, Policy()) * (1+df) / 2);
     }
     else
     {
        result = pow(1 / (1 + basem1), (df + 1) / 2);
     }
     result /= sqrt(df) * boost::math::beta(df / 2, RealType(0.5f), Policy());
   }
   return result;
} // pdf

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType cdf(const students_t_distribution<RealType, Policy>& dist, const RealType& x)
{
   BOOST_MATH_STD_USING // for ADL of std functions

   RealType error_result;
   // degrees_of_freedom > 0 or infinity check:
   RealType df = dist.degrees_of_freedom();
   if (false == detail::check_df_gt0_to_inf(  // Check that df > 0 or == +infinity.
     "boost::math::cdf(const students_t_distribution<%1%>&, %1%)", df, &error_result, Policy()))
   {
     return error_result;
   }
   // Check for bad x first.
   if(false == detail::check_x_not_NaN(
      "boost::math::cdf(const students_t_distribution<%1%>&, %1%)", x, &error_result, Policy()))
   { 
      return error_result;
   }
   if (x == 0)
   { // Special case with exact result.
     return static_cast<RealType>(0.5);
   }
   if ((boost::math::isinf)(x))
   { // x == - or + infinity, regardless of df.
     return ((x < 0) ? static_cast<RealType>(0) : static_cast<RealType>(1));
   }

   RealType limit = policies::get_epsilon<RealType, Policy>();
   // Use policies so that if policy requests lower precision, 
   // then get the normal distribution approximation earlier.
   limit = static_cast<RealType>(1) / limit; // 1/eps
   // for 64-bit double 1/eps = 4503599627370496
   if (df > limit)
   { // Special case for really big degrees_of_freedom > 1 / eps (perhaps infinite?)
     // - use normal distribution which is much faster and more accurate.
     normal_distribution<RealType, Policy> n(0, 1); 
     RealType result = cdf(n, x);
     return result;
   }
   else
   { // normal df case.
     //
     // Calculate probability of Student's t using the incomplete beta function.
     // probability = ibeta(degrees_of_freedom / 2, 1/2, degrees_of_freedom / (degrees_of_freedom + t*t))
     //
     // However when t is small compared to the degrees of freedom, that formula
     // suffers from rounding error, use the identity formula to work around
     // the problem:
     //
     // I[x](a,b) = 1 - I[1-x](b,a)
     //
     // and:
     //
     //     x = df / (df + t^2)
     //
     // so:
     //
     // 1 - x = t^2 / (df + t^2)
     //
     if (fabs(x) > sqrt(tools::max_value<RealType>()))
     {
        // Exact tail values: avoids a spurious intermediate overflow in x * x below.
        return (x < 0) ? static_cast<RealType>(0) : static_cast<RealType>(1);
     }
     RealType x2 = x * x;
     RealType probability;
     if(df > 2 * x2)
     {
        RealType z = x2 / (df + x2);
        probability = ibetac(static_cast<RealType>(0.5), df / 2, z, Policy()) / 2;
     }
     else
     {
        RealType z = df / (df + x2);
        probability = ibeta(df / 2, static_cast<RealType>(0.5), z, Policy()) / 2;
     }
     return (x > 0 ? 1   - probability : probability);
  }
} // cdf

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType quantile(const students_t_distribution<RealType, Policy>& dist, const RealType& p)
{
   BOOST_MATH_STD_USING // for ADL of std functions
   //
   // Obtain parameters:
   RealType probability = p;
 
   // Check for domain errors:
   RealType df = dist.degrees_of_freedom();
   constexpr auto function = "boost::math::quantile(const students_t_distribution<%1%>&, %1%)";
   RealType error_result;
   if(false == (detail::check_df_gt0_to_inf( // Check that df > 0 or == +infinity.
      function, df, &error_result, Policy())
         && detail::check_probability(function, probability, &error_result, Policy())))
      return error_result;
   // Special cases, regardless of degrees_of_freedom.
   if (probability == 0)
      return -policies::raise_overflow_error<RealType>(function, 0, Policy());
   if (probability == 1)
     return policies::raise_overflow_error<RealType>(function, 0, Policy());
   if (probability == static_cast<RealType>(0.5))
     return 0;  //
   //
#if 0
   // This next block is disabled in favour of a faster method than
   // incomplete beta inverse, but code retained for future reference:
   //
   // Calculate quantile of Student's t using the incomplete beta function inverse:
   probability = (probability > 0.5) ? 1 - probability : probability;
   RealType t, x, y;
   x = ibeta_inv(degrees_of_freedom / 2, RealType(0.5), 2 * probability, &y);
   if(degrees_of_freedom * y > tools::max_value<RealType>() * x)
      t = tools::overflow_error<RealType>(function);
   else
      t = sqrt(degrees_of_freedom * y / x);
   //
   // Figure out sign based on the size of p:
   //
   if(p < 0.5)
      t = -t;

   return t;
#endif
   //
   // Depending on how many digits RealType has, this may forward
   // to the incomplete beta inverse as above.  Otherwise uses a
   // faster method that is accurate to ~15 digits everywhere
   // and a couple of epsilon at double precision and in the central 
   // region where most use cases will occur...
   //
   return boost::math::detail::fast_students_t_quantile(df, probability, Policy());
} // quantile

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType cdf(const complemented2_type<students_t_distribution<RealType, Policy>, RealType>& c)
{
   return cdf(c.dist, -c.param);
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType quantile(const complemented2_type<students_t_distribution<RealType, Policy>, RealType>& c)
{
   return -quantile(c.dist, c.param);
}

//
// Parameter estimation follows:
//
namespace detail{

template <class Distribution, class RealType>
BOOST_MATH_GPU_ENABLED inline bool analytical_df_if_cdf_matches(const Distribution& dist, RealType x, RealType p)
{
   RealType cdf_val = cdf(dist, x);
   return epsilon_difference(cdf_val, p) < 4;
}

// Minimum degrees-of-freedom used as the warm-start fallback when the
// Edgeworth approximation yields no valid positive root or is inaccurate
constexpr double df_hint_fallback = 0.01;

//
// Functors for finding degrees of freedom:
//
template <class RealType, class Policy>
struct sample_size_func
{
   BOOST_MATH_GPU_ENABLED sample_size_func(RealType a, RealType b, RealType s, RealType d)
      : alpha(a), beta(b), ratio(s*s/(d*d)) {}

   BOOST_MATH_GPU_ENABLED RealType operator()(const RealType& df)
   {
      if(df <= tools::min_value<RealType>())
      { // 
         return 1;
      }
      students_t_distribution<RealType, Policy> t(df);
      RealType qa, qb;
#ifndef BOOST_MATH_NO_EXCEPTIONS
      try {
#endif
          qa = quantile(complement(t, alpha));
#ifndef BOOST_MATH_NO_EXCEPTIONS
      }
      catch (const std::overflow_error&)
      {
          // Any arbitrary large value will do, must be positive since we calculate qa * qa below:
          return tools::max_value<RealType>();
      }
#endif
#ifndef BOOST_MATH_NO_EXCEPTIONS
      try {
#endif
          qb = quantile(complement(t, beta));
#ifndef BOOST_MATH_NO_EXCEPTIONS
      }
      catch (const std::overflow_error&)
      {
          // Any arbitrary large value will do, will be negative if beta < 0:
          return beta < 0 ? -tools::max_value<RealType>() : tools::max_value<RealType>();
      }
#endif
      qa += qb;
      qa *= qa;
      qa *= ratio;
      qa -= (df + 1);
      return qa;
   }
   RealType alpha, beta, ratio;
};

template <class RealType, class Policy>
struct cdf_to_df_func
{
   BOOST_MATH_GPU_ENABLED cdf_to_df_func(RealType x_val, RealType p_val, bool c)
      : x(x_val), p(p_val), comp(c) {}

   BOOST_MATH_GPU_ENABLED RealType operator()(const RealType& df)
   {
      students_t_distribution<RealType, Policy> t(df);
      return comp ?
         RealType(p - cdf(complement(t, x)))
         : RealType(cdf(t, x) - p);
   }
   RealType x, p;
   bool comp;
};

//
// Shared root-finding helper used by find_degrees_of_freedom.
//
template <class RealType, class Policy, class Func>
BOOST_MATH_GPU_ENABLED RealType solve_for_degrees_of_freedom(
   Func f,
   RealType hint,
   bool rising,
   const char* function,
   Policy const&)
{
   tools::eps_tolerance<RealType> tol(policies::digits<RealType, Policy>());
   boost::math::uintmax_t max_iter = policies::get_max_root_iterations<Policy>();
   boost::math::pair<RealType, RealType> r = tools::bracket_and_solve_root(
      f, hint, RealType(2), rising, tol, max_iter, Policy());
   RealType result = r.first + (r.second - r.first) / 2;
   if (max_iter >= policies::get_max_root_iterations<Policy>())
   {
      return policies::raise_evaluation_error<RealType>(function, // LCOV_EXCL_LINE
         "Unable to locate solution in a reasonable time: either there is no answer to " // LCOV_EXCL_LINE
         "the degrees of freedom or the answer is infinite. Current best guess is %1%", // LCOV_EXCL_LINE
         result, Policy()); // LCOV_EXCL_LINE
   }
   return result;
}

//
// Edgeworth warm-start for find_degrees_of_freedom(t, p).
// On success writes a df estimate into 'result' and returns true.
// Returns false when no positive root is found; 'result' is left unchanged.
//
template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED bool approximate_df_with_edgeworth_expansion(RealType x, RealType p, RealType& result)
{
   BOOST_MATH_STD_USING
   // F(x; nu) ~ cdf_normal(x) - pdf_normal(x)*(x + x^3)/(4*nu) + pdf_normal(x)*(3x + 5x^3 + 7x^5 - 3x^7)/(96*nu^2)
   // Substituting u = 1/nu and setting F(x; nu) = p gives b*u^2 - a*u + c = 0, where:
   //   a = pdf_normal(x) * (x + x^3) / 4
   //   b = pdf_normal(x) * (3x + 5x^3 + 7x^5 - 3x^7) / 96
   //   c = cdf_normal(x) - p
   normal_distribution<RealType, Policy> std_normal(0, 1);
   RealType pdf_val = pdf(std_normal, x);
   RealType cdf_val = cdf(std_normal, x);

   RealType x2 = x * x;
   RealType x3 = x2 * x;
   RealType x5 = x3 * x2;
   RealType x7 = x5 * x2;

   RealType a = pdf_val * (x + x3) / 4;
   RealType b = pdf_val * (3 * x + 5 * x3 + 7 * x5 - 3 * x7) / 96;
   RealType c = cdf_val - p;

   RealType discriminant = a * a - 4 * b * c;
   if (discriminant >= 0 && b != 0)
   {
      RealType sqrt_disc = sqrt(discriminant);
      // The two roots of b*u^2 - a*u + c = 0. Pick the smallest positive u (= largest df = 1/u).
      RealType u = (a - sqrt_disc) / (2 * b);
      if (u <= 0)
         u = (a + sqrt_disc) / (2 * b);
      if (u > 0)
      {
         result = 1 / u;
         return true;
      }
   }
   return false;
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED RealType calculate_hill_df_guess(
   RealType difference_from_mean,
   RealType alpha,
   RealType beta,
   RealType sd)

   // Hill-corrected normal approximation used as an initial guess 
   // for find_degrees_of_freedom(difference_from_mean, alpha, beta, sd).
   // t_p,nu approx z_p * (1 + (z_p^2 + 1) / (4*nu))
   // nu + 1 = (sd/diff)^2 * (t_alpha + t_beta)^2

{
   BOOST_MATH_STD_USING
   normal_distribution<RealType, Policy> n(0, 1);
   RealType za = quantile(complement(n, alpha));
   RealType zb = quantile(complement(n, beta));

   RealType v_norm = pow((za + zb) * sd / difference_from_mean, 2) - 1;
   RealType hint = v_norm + (za * za - za * zb + zb * zb + 1) / 2;

   return (hint <= 0) ? RealType(1) : hint;
}

}  // namespace detail

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED RealType students_t_distribution<RealType, Policy>::find_degrees_of_freedom(
      RealType difference_from_mean,
      RealType alpha,
      RealType beta,
      RealType sd,
      RealType hint)
{
   constexpr auto function = "boost::math::students_t_distribution<%1%>::find_degrees_of_freedom";
   //
   // Check for domain errors:
   //
   RealType error_result;
   if(false == detail::check_probability(
      function, alpha, &error_result, Policy())
         && detail::check_probability(function, beta, &error_result, Policy()))
      return error_result;

   if(hint <= 0)
      hint = detail::calculate_hill_df_guess<RealType, Policy>(difference_from_mean, alpha, beta, sd);

   detail::sample_size_func<RealType, Policy> f(alpha, beta, sd, difference_from_mean);
   return detail::solve_for_degrees_of_freedom(f, hint, false, function, Policy());
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED RealType students_t_distribution<RealType, Policy>::find_degrees_of_freedom(
      RealType t,
      RealType p)
{
   BOOST_MATH_STD_USING // for ADL of std functions
   constexpr auto function = "boost::math::students_t_distribution<%1%>::find_degrees_of_freedom";
   //
   // Check for domain errors:
   //
   RealType error_result;
   if (false == detail::check_probability(function, p, &error_result, Policy()))
      return error_result;

   if (t == 0)
   {
      // CDF(0; df) = 0.5 for all df; only solvable when p == 0.5.
      if (p == static_cast<RealType>(0.5))
         return policies::raise_overflow_error<RealType>(function, nullptr, Policy());
      return policies::raise_domain_error<RealType>(
         function,
         "No degrees of freedom can satisfy CDF(0; df) == %1% (must be 0.5).",
         p, Policy());
   }

   // Analytical cases: df = infinity (normal), df = 1 (cauchy)
   boost::math::normal_distribution<RealType, Policy> norm(0, 1);
   if (detail::analytical_df_if_cdf_matches(norm, t, p))
      return policies::raise_overflow_error<RealType>(function, nullptr, Policy());
   boost::math::cauchy_distribution<RealType, Policy> cauchy(0, 1);
   if (detail::analytical_df_if_cdf_matches(cauchy, t, p))
      return RealType(1);

   // Edgeworth warm start: compute a df estimate; fall back to df_hint_fallback if it fails
   // or is too inaccurate
   RealType hint = static_cast<RealType>(detail::df_hint_fallback);
   if (detail::approximate_df_with_edgeworth_expansion<RealType, Policy>(t, p, hint))
   {
      // Check that approximation is at least somewhat close;
      // for small degrees of freedom it does not fail but is very inaccurate.
      RealType relative_error_threshold = static_cast<RealType>(0.1);
      students_t_distribution<RealType, Policy> t_approx(hint);
      RealType p_approx = cdf(t_approx, t);
      if (relative_difference(p_approx, p) > relative_error_threshold)
         hint = static_cast<RealType>(detail::df_hint_fallback);
   }
   // Root-find on f(df) = CDF(t; df) - p.
   // CDF(t; df) is strictly increasing in df for t > 0, decreasing for t < 0.
   // Use the smaller of p and q=1-p to avoid cancellation near 1.
   RealType q = 1 - p;
   detail::cdf_to_df_func<RealType, Policy> f(t, p < q ? p : q, p < q ? false : true);
   return detail::solve_for_degrees_of_freedom(f, hint, t > 0, function, Policy());
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType mode(const students_t_distribution<RealType, Policy>& /*dist*/)
{
  // Assume no checks on degrees of freedom are useful (unlike mean).
   return 0; // Always zero by definition.
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType median(const students_t_distribution<RealType, Policy>& /*dist*/)
{
   // Assume no checks on degrees of freedom are useful (unlike mean).
   return 0; // Always zero by definition.
}

// See section 5.1 on moments at  http://en.wikipedia.org/wiki/Student%27s_t-distribution

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType mean(const students_t_distribution<RealType, Policy>& dist)
{  // Revised for https://svn.boost.org/trac/boost/ticket/7177
   RealType df = dist.degrees_of_freedom();
   if(((boost::math::isnan)(df)) || (df <= 1) ) 
   { // mean is undefined for moment <= 1!
      return policies::raise_domain_error<RealType>(
      "boost::math::mean(students_t_distribution<%1%> const&, %1%)",
      "Mean is undefined for degrees of freedom < 1 but got %1%.", df, Policy());
      return boost::math::numeric_limits<RealType>::quiet_NaN();
   }
   return 0;
} // mean

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType variance(const students_t_distribution<RealType, Policy>& dist)
{ // http://en.wikipedia.org/wiki/Student%27s_t-distribution
  // Revised for https://svn.boost.org/trac/boost/ticket/7177
  RealType df = dist.degrees_of_freedom();
  if ((boost::math::isnan)(df) || (df <= 2))
  { // NaN or undefined for <= 2.
     return policies::raise_domain_error<RealType>(
      "boost::math::variance(students_t_distribution<%1%> const&, %1%)",
      "variance is undefined for degrees of freedom <= 2, but got %1%.",
      df, Policy());
    return boost::math::numeric_limits<RealType>::quiet_NaN(); // Undefined.
  }
  if ((boost::math::isinf)(df))
  { // +infinity.
    return 1;
  }
  RealType limit = policies::get_epsilon<RealType, Policy>();
  // Use policies so that if policy requests lower precision, 
  // then get the normal distribution approximation earlier.
  limit = static_cast<RealType>(1) / limit; // 1/eps
  // for 64-bit double 1/eps = 4503599627370496
  if (df > limit)
  { // Special case for really big degrees_of_freedom > 1 / eps.
    return 1;
  }
  else
  {
    return df / (df - 2);
  }
} // variance

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType skewness(const students_t_distribution<RealType, Policy>& dist)
{
    RealType df = dist.degrees_of_freedom();
   if( ((boost::math::isnan)(df)) || (dist.degrees_of_freedom() <= 3))
   { // Undefined for moment k = 3.
      return policies::raise_domain_error<RealType>(
         "boost::math::skewness(students_t_distribution<%1%> const&, %1%)",
         "Skewness is undefined for degrees of freedom <= 3, but got %1%.",
         dist.degrees_of_freedom(), Policy());
      return boost::math::numeric_limits<RealType>::quiet_NaN();
   }
   return 0; // For all valid df, including infinity.
} // skewness

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType kurtosis(const students_t_distribution<RealType, Policy>& dist)
{
   RealType df = dist.degrees_of_freedom();
   if(((boost::math::isnan)(df)) || (df <= 4))
   { // Undefined or infinity for moment k = 4.
      return policies::raise_domain_error<RealType>(
       "boost::math::kurtosis(students_t_distribution<%1%> const&, %1%)",
       "Kurtosis is undefined for degrees of freedom <= 4, but got %1%.",
        df, Policy());
        return boost::math::numeric_limits<RealType>::quiet_NaN(); // Undefined.
   }
   if ((boost::math::isinf)(df))
   { // +infinity.
     return 3;
   }
   RealType limit = policies::get_epsilon<RealType, Policy>();
   // Use policies so that if policy requests lower precision, 
   // then get the normal distribution approximation earlier.
   limit = static_cast<RealType>(1) / limit; // 1/eps
   // for 64-bit double 1/eps = 4503599627370496
   if (df > limit)
   { // Special case for really big degrees_of_freedom > 1 / eps.
     return 3;
   }
   else
   {
     //return 3 * (df - 2) / (df - 4); re-arranged to
     return 6 / (df - 4) + 3;
   }
} // kurtosis

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType kurtosis_excess(const students_t_distribution<RealType, Policy>& dist)
{
   // see http://mathworld.wolfram.com/Kurtosis.html

   RealType df = dist.degrees_of_freedom();
   if(((boost::math::isnan)(df)) || (df <= 4))
   { // Undefined or infinity for moment k = 4.
     return policies::raise_domain_error<RealType>(
       "boost::math::kurtosis_excess(students_t_distribution<%1%> const&, %1%)",
       "Kurtosis_excess is undefined for degrees of freedom <= 4, but got %1%.",
      df, Policy());
     return boost::math::numeric_limits<RealType>::quiet_NaN(); // Undefined.
   }
   if ((boost::math::isinf)(df))
   { // +infinity.
     return 0;
   }
   RealType limit = policies::get_epsilon<RealType, Policy>();
   // Use policies so that if policy requests lower precision, 
   // then get the normal distribution approximation earlier.
   limit = static_cast<RealType>(1) / limit; // 1/eps
   // for 64-bit double 1/eps = 4503599627370496
   if (df > limit)
   { // Special case for really big degrees_of_freedom > 1 / eps.
     return 0;
   }
   else
   {
     return 6 / (df - 4);
   }
}

template <class RealType, class Policy>
BOOST_MATH_GPU_ENABLED inline RealType entropy(const students_t_distribution<RealType, Policy>& dist)
{
   BOOST_MATH_STD_USING
   RealType v = dist.degrees_of_freedom();
   RealType vp1 = (v+1)/2;
   RealType vd2 = v/2;

   return vp1*(digamma(vp1) - digamma(vd2)) + log(sqrt(v)*beta(vd2, RealType(1)/RealType(2)));
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

#endif // BOOST_STATS_STUDENTS_T_HPP
