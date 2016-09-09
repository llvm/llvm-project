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
  )

if(WIN32)
  add_definitions(-DGTEST_OS_WINDOWS=1)
endif()

if(SUPPORTS_NO_VARIADIC_MACROS_FLAG)
  add_definitions("-Wno-variadic-macros")
endif()

set(LLVM_REQUIRES_RTTI 1)
add_definitions( -DGTEST_HAS_RTTI=0 )

if (NOT LLVM_ENABLE_THREADS)
  add_definitions( -DGTEST_HAS_PTHREAD=0 )
endif()

find_library(PTHREAD_LIBRARY_PATH pthread)
if (PTHREAD_LIBRARY_PATH)
  list(APPEND LIBS pthread)
endif()

add_llvm_library(gtest
  googletest/src/gtest-all.cc

  LINK_LIBS
  ${LIBS}

  LINK_COMPONENTS
  Support # Depends on llvm::raw_ostream

  # This is a library meant only for the build tree.
  BUILDTREE_ONLY
)

add_subdirectory(UnitTestMain)
