# Copyright 2020 Peter Dimov
# Copyright 2021 Matt Borland
# Distributed under the Boost Software License, Version 1.0.
# https://www.boost.org/LICENSE_1_0.txt

cmake_minimum_required(VERSION 3.5...3.16)

project(boost_math VERSION 1.88.0 LANGUAGES CXX)

add_library(boost_math INTERFACE)

add_library(Boost::math ALIAS boost_math)

target_include_directories(boost_math INTERFACE include)
if(NOT CMAKE_VERSION VERSION_LESS "3.19")
  file(GLOB_RECURSE headers include/*.hpp)
  target_sources(boost_math PRIVATE ${headers})
endif()

include(CMakeDependentOption)

cmake_dependent_option(BOOST_MATH_STANDALONE "Use Boost.Math in standalone mode" ON "NOT BOOST_SUPERPROJECT_VERSION" OFF)

message(STATUS "Boost.Math: standalone mode ${BOOST_MATH_STANDALONE}")

if(BOOST_MATH_STANDALONE)

  target_compile_definitions(boost_math INTERFACE BOOST_MATH_STANDALONE=1)

else()

  target_link_libraries(boost_math
    INTERFACE
      Boost::assert
      Boost::concept_check
      Boost::config
      Boost::core
      Boost::integer
      Boost::lexical_cast
      Boost::predef
      Boost::random
      Boost::static_assert
      Boost::throw_exception
  )

endif()

if(BUILD_TESTING AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/CMakeLists.txt")

  add_subdirectory(test)

# Only enable tests when we're the root project
elseif(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)

  include(CTest)
  add_subdirectory(test)
  
  include(GNUInstallDirs)
  install(DIRECTORY "include/" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

endif()
