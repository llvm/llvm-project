// Check whether the version in CMakeLists.txt is up to date
//
// Copyright 2018 Peter Dimov
//
// Distributed under the Boost Software License, Version 1.0.
//
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt

#include <boost/core/lightweight_test.hpp>
#include <boost/version.hpp>
#include <cstdio>

int main( int ac, char const* av[] )
{
    BOOST_TEST_EQ( ac, 2 );

    if( ac >= 2 )
    {
        char version[ 64 ];
        std::sprintf( version, "%d.%d.%d", BOOST_VERSION / 100000, BOOST_VERSION / 100 % 1000, BOOST_VERSION % 100 );

        BOOST_TEST_CSTR_EQ( av[1], version );
    }

    return boost::report_errors();
}
