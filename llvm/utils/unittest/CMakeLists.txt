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

if(WIN32)
  add_definitions(-DGTEST_OS_WINDOWS=1)
endif()

# Google Test requires headers which need _ALL_SOURCE to build on AIX
if (UNIX AND ${CMAKE_SYSTEM_NAME} MATCHES "AIX")
  remove_definitions("-D_XOPEN_SOURCE=700")
  add_definitions("-D_ALL_SOURCE")
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

# The googletest and googlemock sources don't presently use the 'override'
# keyword, which leads to lots of warnings from -Wsuggest-override. Disable
# that warning here for any targets that link to gtest.
if(CXX_SUPPORTS_SUGGEST_OVERRIDE_FLAG)
  add_definitions("-Wno-suggest-override")
  set_target_properties(gtest PROPERTIES INTERFACE_COMPILE_OPTIONS "-Wno-suggest-override")
endif()

# Gtest 1.8.0 uses tr1/tuple which is deprecated on MSVC, so we force it off.
target_compile_definitions(gtest PUBLIC GTEST_HAS_TR1_TUPLE=0)

if (NOT LLVM_ENABLE_THREADS)
  target_compile_definitions(gtest PUBLIC GTEST_HAS_PTHREAD=0)
endif ()

target_include_directories(gtest
  PUBLIC googletest/include googlemock/include
  PRIVATE googletest googlemock
  )

add_subdirectory(UnitTestMain)

# When LLVM_LINK_LLVM_DYLIB is enabled, libLLVM.so is added to the interface
# link libraries for gtest and gtest_main.  This means that any target, like
# unittests for example, that links against gtest will be forced to link
# against libLLVM.so.  In some cases we may want to statically unittests if they
# need access to symbols that are marked private in libLLVM.so.  The only
# way we can make this work is to remove libLLVM.so from the list of interface
# link libraries for gtest and then make gtest users responsible for explicitly
# adding libLLVM.so to their targets link libraries if they need it.

function (gtest_remove_dylib_from_link_interface target)
  get_target_property(interface_libs ${target} INTERFACE_LINK_LIBRARIES)
  if (interface_libs)
    list(REMOVE_ITEM interface_libs LLVM)
    set_target_properties(${target} PROPERTIES INTERFACE_LINK_LIBRARIES "${interface_libs}")
  endif()
endfunction()

gtest_remove_dylib_from_link_interface(gtest)
gtest_remove_dylib_from_link_interface(gtest_main)
