#define BOOST_TEST_MAIN
// Copyright John Maddock 2007.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/math/policies/policy.hpp>
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

   BOOST_CHECK((std::is_same<normalise<policy<>, digits2<20>, promote_float<false>, discrete_quantile<integer_round_down>, denorm_error<throw_on_error>, domain_error<ignore_error> >::type::domain_error_type, domain_error<ignore_error> >::value));
   BOOST_CHECK((std::is_same<normalise<policy<>, digits2<20>, promote_float<false>, discrete_quantile<integer_round_down>, denorm_error<throw_on_error>, domain_error<ignore_error> >::type::pole_error_type, pole_error<BOOST_MATH_POLE_ERROR_POLICY> >::value));
   BOOST_CHECK((std::is_same<normalise<policy<>, digits2<20>, promote_float<false>, discrete_quantile<integer_round_down>, denorm_error<throw_on_error>, domain_error<ignore_error>  >::type::overflow_error_type, overflow_error<BOOST_MATH_OVERFLOW_ERROR_POLICY> >::value));
   BOOST_CHECK((std::is_same<normalise<policy<>, digits2<20>, promote_float<false>, discrete_quantile<integer_round_down>, denorm_error<throw_on_error>, domain_error<ignore_error>  >::type::underflow_error_type, underflow_error<BOOST_MATH_UNDERFLOW_ERROR_POLICY> >::value));
   BOOST_CHECK((std::is_same<normalise<policy<>, digits2<20>, promote_float<false>, discrete_quantile<integer_round_down>, denorm_error<throw_on_error>, domain_error<ignore_error>  >::type::denorm_error_type, denorm_error<throw_on_error> >::value));
   BOOST_CHECK((std::is_same<normalise<policy<>, digits2<20>, promote_float<false>, discrete_quantile<integer_round_down>, denorm_error<throw_on_error>, domain_error<ignore_error>  >::type::evaluation_error_type, evaluation_error<BOOST_MATH_EVALUATION_ERROR_POLICY> >::value));
   BOOST_CHECK((std::is_same<normalise<policy<>, digits2<20>, promote_float<false>, discrete_quantile<integer_round_down>, denorm_error<throw_on_error>, domain_error<ignore_error>  >::type::indeterminate_result_error_type, indeterminate_result_error<BOOST_MATH_INDETERMINATE_RESULT_ERROR_POLICY> >::value));
   BOOST_CHECK((std::is_same<normalise<policy<>, digits2<20>, promote_float<false>, discrete_quantile<integer_round_down>, denorm_error<throw_on_error>, domain_error<ignore_error>  >::type::precision_type, digits2<20> >::value));
   BOOST_CHECK((std::is_same<normalise<policy<>, digits2<20>, promote_float<false>, discrete_quantile<integer_round_down>, denorm_error<throw_on_error>, domain_error<ignore_error>  >::type::promote_float_type, promote_float<false> >::value));
   BOOST_CHECK((std::is_same<normalise<policy<>, digits2<20>, promote_float<false>, discrete_quantile<integer_round_down>, denorm_error<throw_on_error>, domain_error<ignore_error>  >::type::promote_double_type, policy<>::promote_double_type>::value));
   BOOST_CHECK((std::is_same<normalise<policy<>, digits2<20>, promote_float<false>, discrete_quantile<integer_round_down>, denorm_error<throw_on_error>, domain_error<ignore_error>  >::type::discrete_quantile_type, discrete_quantile<integer_round_down> >::value));

   
} // BOOST_AUTO_TEST_CASE( test_main )



