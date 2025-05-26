//  Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Basic sanity check that header <boost/math/distributions/find_scale.hpp>
// #includes all the files that it needs to.
//
#include <boost/math/distributions/find_scale.hpp>
//
// Note this header includes no other headers, this is
// important if this test is to be meaningful:
//
#include "test_compile_result.hpp"

template <class T, class Policy = boost::math::policies::policy<> >
class test_distribution
{
public:
   typedef T value_type;
   typedef Policy policy_type;
   test_distribution(){}
};

template <class T, class Policy>
T quantile(const test_distribution<T, Policy>&, T)
{
   return 0;
}

template <class T, class Policy>
T quantile(const boost::math::complemented2_type<test_distribution<T, Policy>, T>&)
{
   return 0;
}

namespace boost{ namespace math{ namespace tools{

   template <class T, class Policy> struct is_distribution<test_distribution<T, Policy> > : public std::true_type {};
   template <class T, class Policy> struct is_scaled_distribution<test_distribution<T, Policy> > : public std::true_type {};

}}}

void compile_and_link_test()
{
   check_result<float>(boost::math::find_scale<test_distribution<float> >(f, f, f, boost::math::policies::policy<>()));
   check_result<double>(boost::math::find_scale<test_distribution<double> >(d, d, d, boost::math::policies::policy<>()));
   check_result<long double>(boost::math::find_scale<test_distribution<long double> >(l, l, l, boost::math::policies::policy<>()));

   check_result<float>(boost::math::find_scale<test_distribution<float> >(f, f, f));
   check_result<double>(boost::math::find_scale<test_distribution<double> >(d, d, d));
   check_result<long double>(boost::math::find_scale<test_distribution<long double> >(l, l, l));

   check_result<float>(boost::math::find_scale<test_distribution<float> >(boost::math::complement(f, f, f, boost::math::policies::policy<>())));
   check_result<double>(boost::math::find_scale<test_distribution<double> >(boost::math::complement(d, d, d, boost::math::policies::policy<>())));
   check_result<long double>(boost::math::find_scale<test_distribution<long double> >(boost::math::complement(l, l, l, boost::math::policies::policy<>())));

   check_result<float>(boost::math::find_scale<test_distribution<float> >(boost::math::complement(f, f, f)));
   check_result<double>(boost::math::find_scale<test_distribution<double> >(boost::math::complement(d, d, d)));
   check_result<long double>(boost::math::find_scale<test_distribution<long double> >(boost::math::complement(l, l, l)));
}

