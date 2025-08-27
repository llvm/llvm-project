// Copyright John Maddock 2012.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_TEST_TABLE_TYPE_HPP
#define BOOST_MATH_TEST_TABLE_TYPE_HPP

template <class T>
struct table_type
{
   typedef T type;
};

namespace boost{ namespace math{ namespace concepts{

   class real_concept;

}}}

template <>
struct table_type<boost::math::concepts::real_concept>
{
   typedef long double type;
};

#endif
