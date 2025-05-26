//  Copyright John Maddock 2008.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0.  (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
#include <boost/math/policies/policy.hpp>
#include <boost/math/special_functions/math_fwd.hpp>

namespace boost{ namespace math{ namespace policies{

template <>
struct policy<
   domain_error<errno_on_error>, 
   pole_error<errno_on_error>, 
   overflow_error<errno_on_error>, 
   evaluation_error<errno_on_error>, 
   rounding_error<errno_on_error>, 
   default_policy, default_policy, default_policy, default_policy, default_policy, default_policy>
{
public:
   typedef domain_error<errno_on_error> domain_error_type;
   typedef pole_error<errno_on_error> pole_error_type;
   typedef overflow_error<errno_on_error> overflow_error_type;
   typedef underflow_error<errno_on_error> underflow_error_type;
   typedef denorm_error<errno_on_error> denorm_error_type;
   typedef evaluation_error<errno_on_error> evaluation_error_type;
   typedef rounding_error<errno_on_error> rounding_error_type;
   typedef indeterminate_result_error<> indeterminate_result_error_type;
#if BOOST_MATH_DIGITS10_POLICY == 0
   typedef digits2<> precision_type;
#else
   typedef detail::precision<digits10<>, digits2<> >::type precision_type;
#endif
   typedef promote_float<> promote_float_type;
   typedef promote_double<> promote_double_type;
   typedef discrete_quantile<> discrete_quantile_type;
   typedef assert_undefined<> assert_undefined_type;
   typedef max_series_iterations<> max_series_iterations_type;
   typedef max_root_iterations<> max_root_iterations_type;
};

template <>
struct policy<
   domain_error<errno_on_error>, 
   pole_error<errno_on_error>, 
   overflow_error<errno_on_error>, 
   evaluation_error<errno_on_error>, 
   rounding_error<errno_on_error>, 
   detail::forwarding_arg1, 
   detail::forwarding_arg2, 
   default_policy, default_policy, default_policy, default_policy, default_policy, default_policy>
{
public:
   typedef domain_error<errno_on_error> domain_error_type;
   typedef pole_error<errno_on_error> pole_error_type;
   typedef overflow_error<errno_on_error> overflow_error_type;
   typedef underflow_error<errno_on_error> underflow_error_type;
   typedef denorm_error<errno_on_error> denorm_error_type;
   typedef evaluation_error<errno_on_error> evaluation_error_type;
   typedef rounding_error<errno_on_error> rounding_error_type;
   typedef indeterminate_result_error<> indeterminate_result_error_type;
#if BOOST_MATH_DIGITS10_POLICY == 0
   typedef digits2<> precision_type;
#else
   typedef detail::precision<digits10<>, digits2<> >::type precision_type;
#endif
   typedef promote_float<false> promote_float_type;
   typedef promote_double<false> promote_double_type;
   typedef discrete_quantile<> discrete_quantile_type;
   typedef assert_undefined<> assert_undefined_type;
   typedef max_series_iterations<> max_series_iterations_type;
   typedef max_root_iterations<> max_root_iterations_type;
};

template <>
struct normalise<policy<domain_error<errno_on_error>, pole_error<errno_on_error>, overflow_error<errno_on_error>, evaluation_error<errno_on_error>, rounding_error<errno_on_error> >,
          promote_float<false>,
          promote_double<false>,
          discrete_quantile<>,
          assert_undefined<>,
          default_policy,
          default_policy,
          default_policy,
          default_policy,
          default_policy,
          default_policy,
          default_policy>
{
   typedef policy<domain_error<errno_on_error>, pole_error<errno_on_error>, overflow_error<errno_on_error>, evaluation_error<errno_on_error>, rounding_error<errno_on_error>, detail::forwarding_arg1, detail::forwarding_arg2> type;
};

template <>
struct normalise<policy<domain_error<errno_on_error>, pole_error<errno_on_error>, overflow_error<errno_on_error>, evaluation_error<errno_on_error>, rounding_error<errno_on_error>, detail::forwarding_arg1, detail::forwarding_arg2 >,
          promote_float<false>,
          promote_double<false>,
          discrete_quantile<>,
          assert_undefined<>,
          default_policy,
          default_policy,
          default_policy,
          default_policy,
          default_policy,
          default_policy,
          default_policy>
{
   typedef policy<domain_error<errno_on_error>, pole_error<errno_on_error>, overflow_error<errno_on_error>, evaluation_error<errno_on_error>, rounding_error<errno_on_error>, detail::forwarding_arg1, detail::forwarding_arg2> type;
};

}}} // namespaces

namespace c_policies{

using boost::math::policies::policy;
using boost::math::policies::errno_on_error;
using boost::math::policies::domain_error;
using boost::math::policies::pole_error;
using boost::math::policies::overflow_error;
using boost::math::policies::rounding_error;
using boost::math::policies::evaluation_error;

typedef policy<
   domain_error<errno_on_error>,
   pole_error<errno_on_error>,
   overflow_error<errno_on_error>,
   evaluation_error<errno_on_error>,
   rounding_error<errno_on_error>
> c_policy;

BOOST_MATH_DECLARE_SPECIAL_FUNCTIONS(c_policy)

}
