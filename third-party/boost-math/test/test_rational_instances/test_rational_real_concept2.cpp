//  (C) Copyright John Maddock 2006-7.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include <boost/detail/workaround.hpp>
#if !BOOST_WORKAROUND(BOOST_BORLANDC, BOOST_TESTED_AT(0x582))

#include "test_rational.hpp"
#include <boost/math/concepts/real_concept.hpp>

template void do_test_spots<boost::math::concepts::real_concept, int>(boost::math::concepts::real_concept, int);

#endif
