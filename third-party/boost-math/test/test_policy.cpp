
// Copyright John Maddock 2007.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/math/policies/policy.hpp>
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp> // for test_main
#include <iostream>
#include <type_traits>

template <class P1, class P2>
bool check_same(const P1&, const P2&)
{
   if(!std::is_same<P1, P2>::value)
   {
      std::cout << "P1 = " << typeid(P1).name() << std::endl;
      std::cout << "P2 = " << typeid(P2).name() << std::endl;
   }
   return std::is_same<P1, P2>::value;
}


BOOST_AUTO_TEST_CASE( test_main )
{
   using namespace boost::math::policies;
   using namespace boost;
   BOOST_CHECK(is_domain_error<domain_error<ignore_error> >::value);
   BOOST_CHECK(0 == is_domain_error<pole_error<ignore_error> >::value);
   BOOST_CHECK(is_pole_error<pole_error<ignore_error> >::value);
   BOOST_CHECK(0 == is_pole_error<domain_error<ignore_error> >::value);
   BOOST_CHECK(is_digits10<digits10<ignore_error> >::value);
   BOOST_CHECK(0 == is_digits10<digits2<ignore_error> >::value);

   BOOST_CHECK((std::is_same<policy<>::domain_error_type, domain_error<BOOST_MATH_DOMAIN_ERROR_POLICY> >::value));
   BOOST_CHECK((std::is_same<policy<>::evaluation_error_type, evaluation_error<BOOST_MATH_EVALUATION_ERROR_POLICY> >::value));

   BOOST_CHECK((std::is_same<policy<domain_error<ignore_error> >::domain_error_type, domain_error<ignore_error> >::value));
   BOOST_CHECK((std::is_same<policy<domain_error<ignore_error> >::pole_error_type, pole_error<BOOST_MATH_POLE_ERROR_POLICY> >::value));
   BOOST_CHECK((std::is_same<policy<domain_error<ignore_error> >::overflow_error_type, overflow_error<BOOST_MATH_OVERFLOW_ERROR_POLICY> >::value));
   BOOST_CHECK((std::is_same<policy<domain_error<ignore_error> >::underflow_error_type, underflow_error<BOOST_MATH_UNDERFLOW_ERROR_POLICY> >::value));
   BOOST_CHECK((std::is_same<policy<domain_error<ignore_error> >::denorm_error_type, denorm_error<BOOST_MATH_DENORM_ERROR_POLICY> >::value));
   BOOST_CHECK((std::is_same<policy<domain_error<ignore_error> >::evaluation_error_type, evaluation_error<BOOST_MATH_EVALUATION_ERROR_POLICY> >::value));
   BOOST_CHECK((std::is_same<policy<domain_error<ignore_error> >::indeterminate_result_error_type, indeterminate_result_error<BOOST_MATH_INDETERMINATE_RESULT_ERROR_POLICY> >::value));
   BOOST_CHECK((std::is_same<policy<domain_error<ignore_error> >::precision_type, policy<>::precision_type>::value));
   BOOST_CHECK((std::is_same<policy<domain_error<ignore_error> >::promote_float_type, policy<>::promote_float_type>::value));
   BOOST_CHECK((std::is_same<policy<domain_error<ignore_error> >::promote_double_type, policy<>::promote_double_type>::value));
   BOOST_CHECK((std::is_same<policy<domain_error<ignore_error> >::discrete_quantile_type, policy<>::discrete_quantile_type>::value));

   BOOST_CHECK((std::is_same<policy<pole_error<user_error> >::domain_error_type, policy<>::domain_error_type >::value));
   BOOST_CHECK((std::is_same<policy<pole_error<user_error> >::pole_error_type, pole_error<user_error> >::value));
   BOOST_CHECK((std::is_same<policy<pole_error<user_error> >::overflow_error_type, overflow_error<BOOST_MATH_OVERFLOW_ERROR_POLICY> >::value));
   BOOST_CHECK((std::is_same<policy<pole_error<user_error> >::underflow_error_type, underflow_error<BOOST_MATH_UNDERFLOW_ERROR_POLICY> >::value));
   BOOST_CHECK((std::is_same<policy<pole_error<user_error> >::denorm_error_type, denorm_error<BOOST_MATH_DENORM_ERROR_POLICY> >::value));
   BOOST_CHECK((std::is_same<policy<pole_error<user_error> >::evaluation_error_type, evaluation_error<BOOST_MATH_EVALUATION_ERROR_POLICY> >::value));
   BOOST_CHECK((std::is_same<policy<pole_error<user_error> >::indeterminate_result_error_type, indeterminate_result_error<BOOST_MATH_INDETERMINATE_RESULT_ERROR_POLICY> >::value));
   BOOST_CHECK((std::is_same<policy<pole_error<user_error> >::precision_type, policy<>::precision_type>::value));
   BOOST_CHECK((std::is_same<policy<pole_error<user_error> >::promote_float_type, policy<>::promote_float_type>::value));
   BOOST_CHECK((std::is_same<policy<pole_error<user_error> >::promote_double_type, policy<>::promote_double_type>::value));
   BOOST_CHECK((std::is_same<policy<pole_error<user_error> >::discrete_quantile_type, policy<>::discrete_quantile_type>::value));

   BOOST_CHECK((std::is_same<policy<overflow_error<errno_on_error> >::domain_error_type, policy<>::domain_error_type >::value));
   BOOST_CHECK((std::is_same<policy<overflow_error<errno_on_error> >::pole_error_type, policy<>::pole_error_type >::value));
   BOOST_CHECK((std::is_same<policy<overflow_error<errno_on_error> >::overflow_error_type, overflow_error<errno_on_error> >::value));
   BOOST_CHECK((std::is_same<policy<overflow_error<errno_on_error> >::underflow_error_type, underflow_error<BOOST_MATH_UNDERFLOW_ERROR_POLICY> >::value));
   BOOST_CHECK((std::is_same<policy<overflow_error<errno_on_error> >::denorm_error_type, denorm_error<BOOST_MATH_DENORM_ERROR_POLICY> >::value));
   BOOST_CHECK((std::is_same<policy<overflow_error<errno_on_error> >::evaluation_error_type, evaluation_error<BOOST_MATH_EVALUATION_ERROR_POLICY> >::value));
   BOOST_CHECK((std::is_same<policy<overflow_error<errno_on_error> >::indeterminate_result_error_type, indeterminate_result_error<BOOST_MATH_INDETERMINATE_RESULT_ERROR_POLICY> >::value));
   BOOST_CHECK((std::is_same<policy<overflow_error<errno_on_error> >::precision_type, policy<>::precision_type>::value));
   BOOST_CHECK((std::is_same<policy<overflow_error<errno_on_error> >::promote_float_type, policy<>::promote_float_type>::value));
   BOOST_CHECK((std::is_same<policy<overflow_error<errno_on_error> >::promote_double_type, policy<>::promote_double_type>::value));
   BOOST_CHECK((std::is_same<policy<overflow_error<errno_on_error> >::discrete_quantile_type, policy<>::discrete_quantile_type>::value));

   BOOST_CHECK((std::is_same<policy<underflow_error<errno_on_error> >::domain_error_type, policy<>::domain_error_type >::value));
   BOOST_CHECK((std::is_same<policy<underflow_error<errno_on_error> >::pole_error_type, policy<>::pole_error_type >::value));
   BOOST_CHECK((std::is_same<policy<underflow_error<errno_on_error> >::overflow_error_type, policy<>::overflow_error_type >::value));
   BOOST_CHECK((std::is_same<policy<underflow_error<errno_on_error> >::underflow_error_type, underflow_error<errno_on_error> >::value));
   BOOST_CHECK((std::is_same<policy<underflow_error<errno_on_error> >::denorm_error_type, denorm_error<BOOST_MATH_DENORM_ERROR_POLICY> >::value));
   BOOST_CHECK((std::is_same<policy<underflow_error<errno_on_error> >::evaluation_error_type, evaluation_error<BOOST_MATH_EVALUATION_ERROR_POLICY> >::value));
   BOOST_CHECK((std::is_same<policy<underflow_error<errno_on_error> >::indeterminate_result_error_type, indeterminate_result_error<BOOST_MATH_INDETERMINATE_RESULT_ERROR_POLICY> >::value));
   BOOST_CHECK((std::is_same<policy<underflow_error<errno_on_error> >::precision_type, policy<>::precision_type>::value));
   BOOST_CHECK((std::is_same<policy<underflow_error<errno_on_error> >::promote_float_type, policy<>::promote_float_type>::value));
   BOOST_CHECK((std::is_same<policy<underflow_error<errno_on_error> >::promote_double_type, policy<>::promote_double_type>::value));
   BOOST_CHECK((std::is_same<policy<underflow_error<errno_on_error> >::discrete_quantile_type, policy<>::discrete_quantile_type>::value));

   BOOST_CHECK((std::is_same<policy<denorm_error<errno_on_error> >::domain_error_type, policy<>::domain_error_type >::value));
   BOOST_CHECK((std::is_same<policy<denorm_error<errno_on_error> >::pole_error_type, policy<>::pole_error_type >::value));
   BOOST_CHECK((std::is_same<policy<denorm_error<errno_on_error> >::overflow_error_type, policy<>::overflow_error_type >::value));
   BOOST_CHECK((std::is_same<policy<denorm_error<errno_on_error> >::underflow_error_type, policy<>::underflow_error_type >::value));
   BOOST_CHECK((std::is_same<policy<denorm_error<errno_on_error> >::denorm_error_type, denorm_error<errno_on_error> >::value));
   BOOST_CHECK((std::is_same<policy<denorm_error<errno_on_error> >::evaluation_error_type, evaluation_error<BOOST_MATH_EVALUATION_ERROR_POLICY> >::value));
   BOOST_CHECK((std::is_same<policy<denorm_error<errno_on_error> >::indeterminate_result_error_type, indeterminate_result_error<BOOST_MATH_INDETERMINATE_RESULT_ERROR_POLICY> >::value));
   BOOST_CHECK((std::is_same<policy<denorm_error<errno_on_error> >::precision_type, policy<>::precision_type>::value));
   BOOST_CHECK((std::is_same<policy<denorm_error<errno_on_error> >::promote_float_type, policy<>::promote_float_type>::value));
   BOOST_CHECK((std::is_same<policy<denorm_error<errno_on_error> >::promote_double_type, policy<>::promote_double_type>::value));
   BOOST_CHECK((std::is_same<policy<denorm_error<errno_on_error> >::discrete_quantile_type, policy<>::discrete_quantile_type>::value));

   BOOST_CHECK((std::is_same<policy<evaluation_error<errno_on_error> >::domain_error_type, policy<>::domain_error_type >::value));
   BOOST_CHECK((std::is_same<policy<evaluation_error<errno_on_error> >::pole_error_type, policy<>::pole_error_type >::value));
   BOOST_CHECK((std::is_same<policy<evaluation_error<errno_on_error> >::overflow_error_type, policy<>::overflow_error_type >::value));
   BOOST_CHECK((std::is_same<policy<evaluation_error<errno_on_error> >::underflow_error_type, policy<>::underflow_error_type >::value));
   BOOST_CHECK((std::is_same<policy<evaluation_error<errno_on_error> >::denorm_error_type, policy<>::denorm_error_type >::value));
   BOOST_CHECK((std::is_same<policy<evaluation_error<errno_on_error> >::evaluation_error_type, evaluation_error<errno_on_error> >::value));
   BOOST_CHECK((std::is_same<policy<evaluation_error<errno_on_error> >::indeterminate_result_error_type, policy<>::indeterminate_result_error_type >::value));
   BOOST_CHECK((std::is_same<policy<evaluation_error<errno_on_error> >::precision_type, policy<>::precision_type>::value));
   BOOST_CHECK((std::is_same<policy<evaluation_error<errno_on_error> >::promote_float_type, policy<>::promote_float_type>::value));
   BOOST_CHECK((std::is_same<policy<evaluation_error<errno_on_error> >::promote_double_type, policy<>::promote_double_type>::value));
   BOOST_CHECK((std::is_same<policy<evaluation_error<errno_on_error> >::discrete_quantile_type, policy<>::discrete_quantile_type>::value));

   BOOST_CHECK((std::is_same<policy<indeterminate_result_error<ignore_error> >::domain_error_type, policy<>::domain_error_type >::value));
   BOOST_CHECK((std::is_same<policy<indeterminate_result_error<ignore_error> >::pole_error_type, policy<>::pole_error_type >::value));
   BOOST_CHECK((std::is_same<policy<indeterminate_result_error<ignore_error> >::overflow_error_type, policy<>::overflow_error_type >::value));
   BOOST_CHECK((std::is_same<policy<indeterminate_result_error<ignore_error> >::underflow_error_type, policy<>::underflow_error_type >::value));
   BOOST_CHECK((std::is_same<policy<indeterminate_result_error<ignore_error> >::denorm_error_type, policy<>::denorm_error_type >::value));
   BOOST_CHECK((std::is_same<policy<indeterminate_result_error<ignore_error> >::evaluation_error_type, policy<>::evaluation_error_type >::value));
   BOOST_CHECK((std::is_same<policy<indeterminate_result_error<ignore_error> >::indeterminate_result_error_type, indeterminate_result_error<ignore_error> >::value));
   BOOST_CHECK((std::is_same<policy<indeterminate_result_error<ignore_error> >::precision_type, policy<>::precision_type>::value));
   BOOST_CHECK((std::is_same<policy<indeterminate_result_error<ignore_error> >::promote_float_type, policy<>::promote_float_type>::value));
   BOOST_CHECK((std::is_same<policy<indeterminate_result_error<ignore_error> >::promote_double_type, policy<>::promote_double_type>::value));
   BOOST_CHECK((std::is_same<policy<indeterminate_result_error<ignore_error> >::discrete_quantile_type, policy<>::discrete_quantile_type>::value));

   BOOST_CHECK((std::is_same<policy<digits2<20> >::domain_error_type, policy<>::domain_error_type >::value));
   BOOST_CHECK((std::is_same<policy<digits2<20> >::pole_error_type, policy<>::pole_error_type >::value));
   BOOST_CHECK((std::is_same<policy<digits2<20> >::overflow_error_type, policy<>::overflow_error_type >::value));
   BOOST_CHECK((std::is_same<policy<digits2<20> >::underflow_error_type, policy<>::underflow_error_type >::value));
   BOOST_CHECK((std::is_same<policy<digits2<20> >::denorm_error_type, policy<>::denorm_error_type >::value));
   BOOST_CHECK((std::is_same<policy<digits2<20> >::evaluation_error_type, policy<>::evaluation_error_type >::value));
   BOOST_CHECK((std::is_same<policy<digits2<20> >::indeterminate_result_error_type, policy<>::indeterminate_result_error_type >::value));
   BOOST_CHECK((std::is_same<policy<digits2<20> >::precision_type, digits2<20> >::value));
   BOOST_CHECK((std::is_same<policy<digits2<20> >::promote_float_type, policy<>::promote_float_type>::value));
   BOOST_CHECK((std::is_same<policy<digits2<20> >::promote_double_type, policy<>::promote_double_type>::value));
   BOOST_CHECK((std::is_same<policy<digits2<20> >::discrete_quantile_type, policy<>::discrete_quantile_type>::value));

   BOOST_CHECK((std::is_same<policy<promote_float<false> >::domain_error_type, policy<>::domain_error_type >::value));
   BOOST_CHECK((std::is_same<policy<promote_float<false> >::pole_error_type, policy<>::pole_error_type >::value));
   BOOST_CHECK((std::is_same<policy<promote_float<false> >::overflow_error_type, policy<>::overflow_error_type >::value));
   BOOST_CHECK((std::is_same<policy<promote_float<false> >::underflow_error_type, policy<>::underflow_error_type >::value));
   BOOST_CHECK((std::is_same<policy<promote_float<false> >::denorm_error_type, policy<>::denorm_error_type >::value));
   BOOST_CHECK((std::is_same<policy<promote_float<false> >::evaluation_error_type, policy<>::evaluation_error_type >::value));
   BOOST_CHECK((std::is_same<policy<promote_float<false> >::indeterminate_result_error_type, policy<>::indeterminate_result_error_type >::value));
   BOOST_CHECK((std::is_same<policy<promote_float<false> >::precision_type, policy<>::precision_type >::value));
   BOOST_CHECK((std::is_same<policy<promote_float<false> >::promote_float_type, promote_float<false> >::value));
   BOOST_CHECK((std::is_same<policy<promote_float<false> >::promote_double_type, policy<>::promote_double_type>::value));
   BOOST_CHECK((std::is_same<policy<promote_float<false> >::discrete_quantile_type, policy<>::discrete_quantile_type>::value));

   BOOST_CHECK((std::is_same<policy<promote_double<false> >::domain_error_type, policy<>::domain_error_type >::value));
   BOOST_CHECK((std::is_same<policy<promote_double<false> >::pole_error_type, policy<>::pole_error_type >::value));
   BOOST_CHECK((std::is_same<policy<promote_double<false> >::overflow_error_type, policy<>::overflow_error_type >::value));
   BOOST_CHECK((std::is_same<policy<promote_double<false> >::underflow_error_type, policy<>::underflow_error_type >::value));
   BOOST_CHECK((std::is_same<policy<promote_double<false> >::denorm_error_type, policy<>::denorm_error_type >::value));
   BOOST_CHECK((std::is_same<policy<promote_double<false> >::evaluation_error_type, policy<>::evaluation_error_type >::value));
   BOOST_CHECK((std::is_same<policy<promote_double<false> >::indeterminate_result_error_type, policy<>::indeterminate_result_error_type >::value));
   BOOST_CHECK((std::is_same<policy<promote_double<false> >::precision_type, policy<>::precision_type >::value));
   BOOST_CHECK((std::is_same<policy<promote_double<false> >::promote_float_type,  policy<>::promote_float_type>::value));
   BOOST_CHECK((std::is_same<policy<promote_double<false> >::promote_double_type, promote_double<false> >::value));
   BOOST_CHECK((std::is_same<policy<promote_double<false> >::discrete_quantile_type, policy<>::discrete_quantile_type>::value));

   BOOST_CHECK((std::is_same<policy<discrete_quantile<integer_round_up> >::domain_error_type, policy<>::domain_error_type >::value));
   BOOST_CHECK((std::is_same<policy<discrete_quantile<integer_round_up> >::pole_error_type, policy<>::pole_error_type >::value));
   BOOST_CHECK((std::is_same<policy<discrete_quantile<integer_round_up> >::overflow_error_type, policy<>::overflow_error_type >::value));
   BOOST_CHECK((std::is_same<policy<discrete_quantile<integer_round_up> >::underflow_error_type, policy<>::underflow_error_type >::value));
   BOOST_CHECK((std::is_same<policy<discrete_quantile<integer_round_up> >::denorm_error_type, policy<>::denorm_error_type >::value));
   BOOST_CHECK((std::is_same<policy<discrete_quantile<integer_round_up> >::evaluation_error_type, policy<>::evaluation_error_type >::value));
   BOOST_CHECK((std::is_same<policy<discrete_quantile<integer_round_up> >::indeterminate_result_error_type, policy<>::indeterminate_result_error_type >::value));
   BOOST_CHECK((std::is_same<policy<discrete_quantile<integer_round_up> >::precision_type, policy<>::precision_type >::value));
   BOOST_CHECK((std::is_same<policy<discrete_quantile<integer_round_up> >::promote_float_type,  policy<>::promote_float_type>::value));
   BOOST_CHECK((std::is_same<policy<discrete_quantile<integer_round_up> >::promote_double_type, policy<>::promote_double_type>::value));
   BOOST_CHECK((std::is_same<policy<discrete_quantile<integer_round_up> >::discrete_quantile_type, discrete_quantile<integer_round_up> >::value));
   
} // BOOST_AUTO_TEST_CASE( test_main )



