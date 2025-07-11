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

   BOOST_CHECK(check_same(make_policy(domain_error<ignore_error>()), policy<domain_error<ignore_error> >()));
   BOOST_CHECK(check_same(make_policy(domain_error<ignore_error>(), pole_error<ignore_error>()), policy<domain_error<ignore_error>, pole_error<ignore_error> >()));
   BOOST_CHECK(check_same(make_policy(domain_error<ignore_error>(), pole_error<ignore_error>(), overflow_error<ignore_error>()), policy<domain_error<ignore_error>, pole_error<ignore_error>, overflow_error<ignore_error> >()));
   BOOST_CHECK(check_same(make_policy(domain_error<ignore_error>(), pole_error<ignore_error>(), overflow_error<ignore_error>(), underflow_error<throw_on_error>()), policy<domain_error<ignore_error>, pole_error<ignore_error>, overflow_error<ignore_error>, underflow_error<throw_on_error> >()));
   BOOST_CHECK(check_same(make_policy(domain_error<ignore_error>(), pole_error<ignore_error>(), overflow_error<ignore_error>(), underflow_error<throw_on_error>(), denorm_error<throw_on_error>()), policy<domain_error<ignore_error>, pole_error<ignore_error>, overflow_error<ignore_error>, underflow_error<throw_on_error>, denorm_error<throw_on_error> >()));
   BOOST_CHECK(check_same(make_policy(domain_error<ignore_error>(), pole_error<ignore_error>(), overflow_error<ignore_error>(), underflow_error<throw_on_error>(), denorm_error<throw_on_error>(), evaluation_error<ignore_error>()), policy<domain_error<ignore_error>, pole_error<ignore_error>, overflow_error<ignore_error>, underflow_error<throw_on_error>, denorm_error<throw_on_error>, evaluation_error<ignore_error> >()));
   BOOST_CHECK(check_same(make_policy(domain_error<ignore_error>(), pole_error<ignore_error>(), overflow_error<ignore_error>(), underflow_error<throw_on_error>(), denorm_error<throw_on_error>(), evaluation_error<ignore_error>(), indeterminate_result_error<throw_on_error>()), policy<domain_error<ignore_error>, pole_error<ignore_error>, overflow_error<ignore_error>, underflow_error<throw_on_error>, denorm_error<throw_on_error>, evaluation_error<ignore_error>, indeterminate_result_error<throw_on_error> >()));
   BOOST_CHECK(check_same(make_policy(domain_error<ignore_error>(), pole_error<ignore_error>(), overflow_error<ignore_error>(), underflow_error<throw_on_error>(), denorm_error<throw_on_error>(), evaluation_error<ignore_error>(), indeterminate_result_error<throw_on_error>(), digits2<10>()), policy<domain_error<ignore_error>, pole_error<ignore_error>, overflow_error<ignore_error>, underflow_error<throw_on_error>, denorm_error<throw_on_error>, evaluation_error<ignore_error>, indeterminate_result_error<throw_on_error>, digits2<10> >()));
   BOOST_CHECK(check_same(make_policy(domain_error<ignore_error>(), pole_error<ignore_error>(), overflow_error<ignore_error>(), underflow_error<throw_on_error>(), denorm_error<throw_on_error>(), evaluation_error<ignore_error>(), indeterminate_result_error<throw_on_error>(), digits10<5>()), policy<domain_error<ignore_error>, pole_error<ignore_error>, overflow_error<ignore_error>, underflow_error<throw_on_error>, denorm_error<throw_on_error>, evaluation_error<ignore_error>, indeterminate_result_error<throw_on_error>, digits2<19> >()));
   BOOST_CHECK(check_same(make_policy(domain_error<ignore_error>(), pole_error<ignore_error>(), overflow_error<ignore_error>(), underflow_error<throw_on_error>(), denorm_error<throw_on_error>(), evaluation_error<ignore_error>(), indeterminate_result_error<throw_on_error>(), digits2<10>(), promote_float<false>()), policy<domain_error<ignore_error>, pole_error<ignore_error>, overflow_error<ignore_error>, underflow_error<throw_on_error>, denorm_error<throw_on_error>, evaluation_error<ignore_error>, indeterminate_result_error<throw_on_error>, digits2<10>, promote_float<false> >()));
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   BOOST_CHECK(check_same(make_policy(domain_error<ignore_error>(), pole_error<ignore_error>(), overflow_error<ignore_error>(), underflow_error<throw_on_error>(), denorm_error<throw_on_error>(), evaluation_error<ignore_error>(), indeterminate_result_error<throw_on_error>(), digits2<10>(), promote_float<false>(), promote_double<false>()), policy<domain_error<ignore_error>, pole_error<ignore_error>, overflow_error<ignore_error>, underflow_error<throw_on_error>, denorm_error<throw_on_error>, evaluation_error<ignore_error>, indeterminate_result_error<throw_on_error>, digits2<10>, promote_float<false>, promote_double<false> >()));
   BOOST_CHECK(check_same(make_policy(domain_error<ignore_error>(), pole_error<ignore_error>(), overflow_error<ignore_error>(), underflow_error<throw_on_error>(), denorm_error<throw_on_error>(), evaluation_error<ignore_error>(), indeterminate_result_error<throw_on_error>(), digits2<10>(), promote_float<false>(), promote_double<false>(), discrete_quantile<integer_round_down>()), policy<domain_error<ignore_error>, pole_error<ignore_error>, overflow_error<ignore_error>, underflow_error<throw_on_error>, denorm_error<throw_on_error>, evaluation_error<ignore_error>, indeterminate_result_error<throw_on_error>, digits2<10>, promote_float<false>, promote_double<false>, discrete_quantile<integer_round_down> >()));
#endif

   
} // BOOST_AUTO_TEST_CASE( test_main )



