# Copyright 2021 Matt Borland
# Distributed under the Boost Software License, Version 1.0.
# https://www.boost.org/LICENSE_1_0.txt

file(GLOB SOURCES "*.cpp")
add_library(examples ${SOURCES})
target_compile_features(examples PRIVATE cxx_std_17)
target_include_directories(compile_tests PUBLIC ${CMAKE_SOURCE_DIR}/include)
