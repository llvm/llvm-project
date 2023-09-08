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

if (HAVE_LIBPTHREAD)
  list(APPEND LIBS pthread)
endif()

# Do not build unittest libraries automatically, they will be pulled in
# by unittests if these are built.

set(BUILDTREE_ONLY BUILDTREE_ONLY)
set(EXCLUDE_FROM_ALL ON)
if (LLVM_INSTALL_GTEST)
  set(EXCLUDE_FROM_ALL OFF)
  set(BUILDTREE_ONLY "")
endif ()

add_llvm_library(llvm_gtest
  googletest/src/gtest-all.cc
  googlemock/src/gmock-all.cc

  LINK_LIBS
  ${LIBS}

  LINK_COMPONENTS
  Support # Depends on llvm::raw_ostream

  # This is a library meant only for the build tree.
  ${BUILDTREE_ONLY}
)

# The googletest and googlemock sources don't presently use the 'override'
# keyword, which leads to lots of warnings from -Wsuggest-override. Disable
# that warning here for any targets that link to gtest.
if(CXX_SUPPORTS_SUGGEST_OVERRIDE_FLAG)
  add_definitions("-Wno-suggest-override")
  set_target_properties(llvm_gtest PROPERTIES INTERFACE_COMPILE_OPTIONS "-Wno-suggest-override")
endif()

if (NOT LLVM_ENABLE_THREADS)
  target_compile_definitions(llvm_gtest PUBLIC GTEST_HAS_PTHREAD=0)
endif ()

target_include_directories(llvm_gtest
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/googletest/include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/googlemock/include>
         $<INSTALL_INTERFACE:include/llvm-gtest/>
         $<INSTALL_INTERFACE:include/llvm-gmock/>
  PRIVATE googletest googlemock
  )

add_subdirectory(UnitTestMain)

if (LLVM_INSTALL_GTEST)
  install(TARGETS llvm_gtest llvm_gtest_main LLVMTestingSupport LLVMTestingAnnotations
	  ARCHIVE DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT llvm_gtest)
  install(DIRECTORY googletest/include/gtest/ DESTINATION include/llvm-gtest/gtest/ COMPONENT llvm_gtest)
  install(DIRECTORY googlemock/include/gmock/ DESTINATION include/llvm-gmock/gmock/ COMPONENT llvm_gtest)
endif()

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

gtest_remove_dylib_from_link_interface(llvm_gtest)
gtest_remove_dylib_from_link_interface(llvm_gtest_main)
