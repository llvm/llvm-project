# Copyright 2021 Matt Borland
# Distributed under the Boost Software License, Version 1.0.
# https://www.boost.org/LICENSE_1_0.txt

include_directories(../../include_private)
file(GLOB SRC_FILES CONFIGURE_DEPENDS "*.cpp")
boost_test(TYPE "compile" SOURCES ${SRC_FILES} COMPILE_DEFINITIONS BOOST_MATH_STANDALONE COMPILE_FEATURES cxx_std_17 LINK_LIBRARIES Boost::math Boost::multiprecision Boost::numeric_ublas Boost::unit_test_framework )
