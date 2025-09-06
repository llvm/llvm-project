
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

   BOOST_CHECK(check_same(make_policy(), policy<>()));
   BOOST_CHECK(check_same(make_policy(denorm_error<ignore_error>()), normalise<policy<denorm_error<ignore_error> > >::type()));
   BOOST_CHECK(check_same(make_policy(digits2<20>()), normalise<policy<digits2<20> > >::type()));
   BOOST_CHECK(check_same(make_policy(promote_float<false>()), normalise<policy<promote_float<false> > >::type()));
   BOOST_CHECK(check_same(make_policy(domain_error<ignore_error>()), normalise<policy<domain_error<ignore_error> > >::type()));
   BOOST_CHECK(check_same(make_policy(pole_error<ignore_error>()), normalise<policy<pole_error<ignore_error> > >::type()));
   BOOST_CHECK(check_same(make_policy(indeterminate_result_error<ignore_error>()), normalise<policy<indeterminate_result_error<ignore_error> > >::type()));

   
} // BOOST_AUTO_TEST_CASE( test_main )



