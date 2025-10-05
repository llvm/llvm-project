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

   BOOST_CHECK((std::is_same<policy<domain_error<ignore_error>, overflow_error<ignore_error> >::domain_error_type, domain_error<ignore_error> >::value));
   BOOST_CHECK((std::is_same<policy<domain_error<ignore_error>, overflow_error<ignore_error> >::pole_error_type, pole_error<BOOST_MATH_POLE_ERROR_POLICY> >::value));
   BOOST_CHECK((std::is_same<policy<domain_error<ignore_error>, overflow_error<ignore_error> >::overflow_error_type, overflow_error<ignore_error> >::value));
   BOOST_CHECK((std::is_same<policy<domain_error<ignore_error>, overflow_error<ignore_error> >::underflow_error_type, underflow_error<BOOST_MATH_UNDERFLOW_ERROR_POLICY> >::value));
   BOOST_CHECK((std::is_same<policy<domain_error<ignore_error>, overflow_error<ignore_error> >::denorm_error_type, denorm_error<BOOST_MATH_DENORM_ERROR_POLICY> >::value));
   BOOST_CHECK((std::is_same<policy<domain_error<ignore_error>, overflow_error<ignore_error> >::evaluation_error_type, evaluation_error<BOOST_MATH_EVALUATION_ERROR_POLICY> >::value));
   BOOST_CHECK((std::is_same<policy<domain_error<ignore_error>, overflow_error<ignore_error> >::indeterminate_result_error_type, indeterminate_result_error<BOOST_MATH_INDETERMINATE_RESULT_ERROR_POLICY> >::value));
   BOOST_CHECK((std::is_same<policy<domain_error<ignore_error>, overflow_error<ignore_error> >::precision_type, policy<>::precision_type>::value));
   BOOST_CHECK((std::is_same<policy<domain_error<ignore_error>, overflow_error<ignore_error> >::promote_float_type, policy<>::promote_float_type>::value));
   BOOST_CHECK((std::is_same<policy<domain_error<ignore_error>, overflow_error<ignore_error> >::promote_double_type, policy<>::promote_double_type>::value));
   BOOST_CHECK((std::is_same<policy<domain_error<ignore_error>, overflow_error<ignore_error> >::discrete_quantile_type, policy<>::discrete_quantile_type>::value));

   
} // BOOST_AUTO_TEST_CASE( test_main )



