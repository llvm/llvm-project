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

set(LLVM_SUBPROJECT_TITLE "Third-Party/Google Test")

if(WIN32)
  add_definitions(-DGTEST_OS_WINDOWS=1)
endif()

# Google Test requires headers which need _ALL_SOURCE to build on AIX
if (UNIX AND "${CMAKE_SYSTEM_NAME}" MATCHES "AIX")
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

# Install GTest only if requested.
set(BUILDTREE_ONLY BUILDTREE_ONLY)
if (LLVM_INSTALL_GTEST)
  set(BUILDTREE_ONLY "")
endif ()

if (LLVM_RUNTIMES_BUILD)
  # We cannot use llvm_gtest for the target; it would clash with
  # find_package(LLVM) with LLVM_EXPORT_GTEST=ON. Instead, we define an alias
  # default_gtest that points to llvm_gtest in the LLVM build and
  # runtimes_gtest in an runtimes build.
  set(gtest_name "runtimes_gtest")

  # This instance of GTest is use for unittests for the runtime libraries. It
  # must not link to LLVMSupport (used for llvm::raw_ostream and llvm::cl
  # support) of the host build: It may be a different architecture and/or
  # using a different C++ ABI such as libcxx/libstdc++.
  add_library("${gtest_name}" STATIC
    googletest/src/gtest-all.cc
    googlemock/src/gmock-all.cc
  )
  target_compile_definitions("${gtest_name}" PUBLIC GTEST_NO_LLVM_SUPPORT=1)
  target_link_libraries("${gtest_name}"
    PUBLIC # STATIC -OR- PRIVATE for SHARED
      ${LIBS}
  )

  add_library("${gtest_name}_main" STATIC
    googletest/src/gtest_main.cc
  )
  target_link_libraries("${gtest_name}_main"
    PUBLIC
      ${gtest_name}
  )
else()
  set(gtest_name "llvm_gtest")

  add_llvm_library("${gtest_name}"
    googletest/src/gtest-all.cc
    googlemock/src/gmock-all.cc

    LINK_LIBS
    ${LIBS}

    LINK_COMPONENTS
    Support

    # This is a library meant only for the build tree.
    ${BUILDTREE_ONLY}
  )

  # When used from the buildtree, also force use of buildtree LLVM headers,
  # (instead locally installed version)
  # FIXME: Shouldn't this be done for all LLVM libraries? Currently, LLVM uses a
  # big giant `include_directories( ${LLVM_INCLUDE_DIR} ${LLVM_MAIN_INCLUDE_DIR})`
  # which CMake does not add to the import library.
  target_include_directories("${gtest_name}" BEFORE
    PUBLIC $<BUILD_INTERFACE:${LLVM_SOURCE_DIR}/include>
           $<BUILD_INTERFACE:${LLVM_BINARY_DIR}/include>
  )

  add_llvm_library("${gtest_name}_main"
    UnitTestMain/TestMain.cpp

    LINK_LIBS
    "${gtest_name}"

    LINK_COMPONENTS
    Support

    ${BUILDTREE_ONLY}
  )

  install(DIRECTORY googletest/include/gtest/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/llvm-gtest/gtest/" COMPONENT "${gtest_name}")
  install(DIRECTORY googlemock/include/gmock/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/llvm-gmock/gmock/" COMPONENT "${gtest_name}")
endif()

# The googletest and googlemock sources don't presently use the 'override'
# keyword, which leads to lots of warnings from -Wsuggest-override. Disable
# that warning here for any targets that link to gtest.
if(CXX_SUPPORTS_SUGGEST_OVERRIDE_FLAG)
  add_definitions("-Wno-suggest-override")
  set_target_properties("${gtest_name}" PROPERTIES INTERFACE_COMPILE_OPTIONS "-Wno-suggest-override")
endif()

if (NOT LLVM_ENABLE_THREADS)
  target_compile_definitions("${gtest_name}" PUBLIC GTEST_HAS_PTHREAD=0)
endif ()

target_include_directories("${gtest_name}"
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/googletest/include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/googlemock/include>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/llvm-gtest/>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/llvm-gmock/>
  PRIVATE googletest googlemock
  )

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

if (NOT LLVM_RUNTIMES_BUILD)
  gtest_remove_dylib_from_link_interface("${gtest_name}")
  gtest_remove_dylib_from_link_interface("${gtest_name}_main")
endif ()

# The build processing this file always uses this GTest for unittests.
# Projects that do not use the LLVM_ENABLE_PROJECTS mechanism, but are
# standalone-builds using find_package(LLVM) can define these aliases
# explicitly.
add_library(default_gtest ALIAS "${gtest_name}")
add_library(default_gtest_main ALIAS "${gtest_name}_main")
