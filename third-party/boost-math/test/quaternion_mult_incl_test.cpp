// test file for quaternion.hpp

//  (C) Copyright Hubert Holin 2001.
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)


#include <boost/test/unit_test.hpp>
#include <boost/test/unit_test_log.hpp>

#include "quaternion_mi1.h"
#include "quaternion_mi2.h"


boost::unit_test::test_suite *    init_unit_test_suite(int, char *[])
{
    ::boost::unit_test::unit_test_log.
        set_threshold_level(::boost::unit_test::log_messages);
    
    boost::unit_test::test_suite *    test =
        BOOST_TEST_SUITE("quaternion_multiple_inclusion_test");
    
    BOOST_TEST_MESSAGE("Results of quaternion (multiple inclusion) test.");
    BOOST_TEST_MESSAGE(" ");
    BOOST_TEST_MESSAGE("(C) Copyright Hubert Holin 2003-2005.");
    BOOST_TEST_MESSAGE("Distributed under the Boost Software License, Version 1.0.");
    BOOST_TEST_MESSAGE("(See accompanying file LICENSE_1_0.txt or copy at");
    BOOST_TEST_MESSAGE("http://www.boost.org/LICENSE_1_0.txt)");
    BOOST_TEST_MESSAGE(" ");
    
    test->add(BOOST_TEST_CASE(&quaternion_mi1));
    test->add(BOOST_TEST_CASE(&quaternion_mi2));
    
    return(test);
}

