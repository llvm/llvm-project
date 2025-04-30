cmake_minimum_required(VERSION 3.5)

project(next-ir-lib)

if(NOT CMAKE_BUILD_TYPE)
    message(WARNING "Defaulting to debug build. To specify build type: cmake -DCMAKE_BUILD_TYPE=[Release|Debug|RelWithDebInfo]")
    set(CMAKE_BUILD_TYPE Debug)
endif()

# TODO: Check neccessary compile options, we build this with flatcc.

# Set NEXT_IR_LIBDIR_SUBDIR to next-ir-lib
set(NEXT_IR_LIBDIR_SUBDIR next-ir-lib CACHE STRING "")

set(NEXT_IR_LIBRARY_DIR ${CMAKE_BINARY_DIR}/lib/${NEXT_IR_LIBDIR_SUBDIR} CACHE STRING "")
set(NEXT_IR_INSTALL_LIBRARY_DIR lib/${NEXT_IR_LIBDIR_SUBDIR} CACHE STRING "")

add_subdirectory(lib)
