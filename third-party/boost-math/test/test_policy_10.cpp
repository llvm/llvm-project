
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

   BOOST_CHECK(check_same(make_policy(denorm_error<ignore_error>(), digits2<20>()), make_policy(digits2<20>(), denorm_error<ignore_error>())));
   BOOST_CHECK(check_same(make_policy(denorm_error<ignore_error>(), promote_float<false>()), make_policy(promote_float<false>(), denorm_error<ignore_error>())));
   BOOST_CHECK(check_same(make_policy(denorm_error<ignore_error>(), indeterminate_result_error<ignore_error>(), promote_float<false>()), make_policy(indeterminate_result_error<ignore_error>(), promote_float<false>(), denorm_error<ignore_error>())));
} // BOOST_AUTO_TEST_CASE( test_main )



