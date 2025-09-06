//  (C) Copyright John Maddock 2006-7.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include "test_rational.hpp"

#ifdef BOOST_HAS_LONG_LONG
template void do_test_spots<float, boost::ulong_long_type>(float, boost::ulong_long_type);
#endif
