########################################################################
# Experimental CMake build script for Google Test.
#
# Consider this a prototype.  It will change drastically.  For now,
# this is only for people on the cutting edge.
#
# To run the tests for Google Test itself on Linux, use 'make test' or
# ctest.  You can select which tests to run using 'ctest -R regex'.
# For more options, run 'ctest --help'.
########################################################################
#
# Project-wide settings

# Where gtest's .h files can be found.
include_directories(
  googletest/include
  googletest
  googlemock/include
  googlemock
  )

# LLVM requires C++11 but gtest doesn't correctly detect the availability
# of C++11 on MSVC, so we force it on.
add_definitions(-DGTEST_LANG_CXX11=1)
add_definitions(-DGTEST_HAS_TR1_TUPLE=0)

if(WIN32)
  add_definitions(-DGTEST_OS_WINDOWS=1)
endif()

if(SUPPORTS_VARIADIC_MACROS_FLAG)
  add_definitions("-Wno-variadic-macros")
endif()
if(SUPPORTS_GNU_ZERO_VARIADIC_MACRO_ARGUMENTS_FLAG)
  add_definitions("-Wno-gnu-zero-variadic-macro-arguments")
endif()
if(CXX_SUPPORTS_COVERED_SWITCH_DEFAULT_FLAG)
  add_definitions("-Wno-covered-switch-default")
endif()

set(LLVM_REQUIRES_RTTI 1)
add_definitions( -DGTEST_HAS_RTTI=0 )

if (NOT LLVM_ENABLE_THREADS)
  add_definitions( -DGTEST_HAS_PTHREAD=0 )
endif()

find_library(LLVM_PTHREAD_LIBRARY_PATH pthread)
if (LLVM_PTHREAD_LIBRARY_PATH)
  list(APPEND LIBS pthread)
endif()

add_llvm_library(gtest
  googletest/src/gtest-all.cc
  googlemock/src/gmock-all.cc

  LINK_LIBS
  ${LIBS}

  LINK_COMPONENTS
  Support # Depends on llvm::raw_ostream

  # This is a library meant only for the build tree.
  BUILDTREE_ONLY
)

add_subdirectory(UnitTestMain)
