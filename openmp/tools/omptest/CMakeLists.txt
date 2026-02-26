##===----------------------------------------------------------------------===##
#
# Build OMPT unit testing library: ompTest
#
##===----------------------------------------------------------------------===##

cmake_minimum_required(VERSION 3.20)
project(omptest LANGUAGES CXX)

option(LIBOMPTEST_BUILD_STANDALONE
       "Build ompTest 'standalone', i.e. w/o GoogleTest." OFF)
option(LIBOMPTEST_BUILD_UNITTESTS
       "Build ompTest's unit tests, requires GoogleTest." OFF)
option(LIBOMPTEST_INSTALL_COMPONENTS
       "Install ompTest library, headers and package files." OFF)

# Exit early if OMPT support or LLVM-tests were disabled by the user.
if((NOT ${LIBOMP_OMPT_SUPPORT}) OR (NOT ${LLVM_INCLUDE_TESTS}))
  return()
endif()

# Only support OpenMP runtime 'bootstrap' build mode.
# Check as seen in 'runtimes/CMakeLists.txt', due to passing HAVE_LLVM_LIT to
# llvm_ExternalProject_Add() only.
if (NOT HAVE_LLVM_LIT)
  message(STATUS "Skipping omptest build. Reason: Only supported in bootstrap"
                 " build mode of OpenMP runtime.")
  return()
endif()

include(CMakePackageConfigHelpers)

include_directories(${LIBOMP_INCLUDE_DIR})

set(OMPTEST_HEADERS
  ./include/AssertMacros.h
  ./include/InternalEvent.h
  ./include/InternalEventCommon.h
  ./include/Logging.h
  ./include/OmptAliases.h
  ./include/OmptAsserter.h
  ./include/OmptAssertEvent.h
  ./include/OmptCallbackHandler.h
  ./include/OmptTester.h
  ./include/OmptTesterGlobals.h
)

add_library(omptest
  SHARED

  ${OMPTEST_HEADERS}
  ./src/InternalEvent.cpp
  ./src/InternalEventOperators.cpp
  ./src/Logging.cpp
  ./src/OmptAsserter.cpp
  ./src/OmptAssertEvent.cpp
  ./src/OmptCallbackHandler.cpp
  ./src/OmptTester.cpp
)

# Target: ompTest library
# On (implicit) request of GoogleTest, embed the sources provided with LLVM.
if ((NOT LIBOMPTEST_BUILD_STANDALONE) OR LIBOMPTEST_BUILD_UNITTESTS)
  # Check if standalone build was requested together with unittests
  if (LIBOMPTEST_BUILD_STANDALONE)
    # Emit warning: this build actually depends on LLVM's GoogleTest
    message(WARNING "LIBOMPTEST_BUILD_STANDALONE and LIBOMPTEST_BUILD_UNITTESTS"
                    " requested simultaneously.\n"
                    "Linking against LLVM's GoogleTest library archives.\n"
                    "Disable LIBOMPTEST_BUILD_UNITTESTS to perform an actual"
                    " standalone build.")
    # Explicitly disable LIBOMPTEST_BUILD_STANDALONE
    set(LIBOMPTEST_BUILD_STANDALONE OFF)
  endif()

  message(STATUS "omptest build mode: GTest-based")

  # Add GoogleTest-based header and embed GTest symbols into the shared lib.
  # Merging of GTest is desired, such that omptest is self-contained and
  # independent of external GTest installations.
  target_sources(omptest PRIVATE
    $<TARGET_OBJECTS:default_gtest>
  )

  # Link against the default GTest which at this point primarily pulls in the
  # include directories and compile definitions. It is important to make these
  # available to dependant targets, e.g. for unit tests.
  target_link_libraries(omptest PUBLIC default_gtest)

  # Link against Threads (recommended for GTest).
  find_package(Threads REQUIRED)
  target_link_libraries(omptest PUBLIC Threads::Threads)

  # Ensure that embedded GTest symbols are exported from libomptest.so even in
  # builds that default to hidden.
  set_target_properties(omptest PROPERTIES
    CXX_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN OFF
  )
else()
  message(STATUS "omptest build mode: standalone")

  # Add 'standalone' compile definitions
  target_compile_definitions(omptest PRIVATE
    -DOPENMP_LIBOMPTEST_BUILD_STANDALONE)

  # Add 'standalone' source files
  target_sources(omptest PRIVATE
    ./src/OmptTesterStandalone.cpp)
endif()

if(TARGET cxx-headers)
  add_dependencies(omptest cxx-headers)
endif()

if(TARGET cxx_shared)
  add_dependencies(omptest cxx_shared)
endif()

if(TARGET cxxabi_shared)
  add_dependencies(omptest cxxabi_shared)
endif()

# Add common include directories.
target_include_directories(omptest PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${LIBOMP_HEADERS_INSTALL_PATH}/omptest>
)

target_compile_features(omptest PRIVATE cxx_std_17)

# Perform package configuration.
set(OMPTEST_CONFIG_INSTALL_DIR "${OPENMP_INSTALL_LIBDIR}/cmake/openmp/omptest")
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/omptest-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/cmake/omptest-config.cmake
  INSTALL_DESTINATION "${OMPTEST_CONFIG_INSTALL_DIR}"
)

# Perform installation only if requested by the user.
if(LIBOMPTEST_INSTALL_COMPONENTS)
  # Install package configuration files.
  install(FILES ${omptest_BINARY_DIR}/cmake/omptest-config.cmake
          DESTINATION "${OMPTEST_CONFIG_INSTALL_DIR}")

  # Install libomptest header files: Copy header-files from include dir
  set(OMPTEST_HEADER_INSTALL_DIR "${LIBOMP_HEADERS_INSTALL_PATH}/omptest")
  install(DIRECTORY ./include/
          DESTINATION "${OMPTEST_HEADER_INSTALL_DIR}"
          FILES_MATCHING PATTERN "*.h")

  # Install library and export targets.
  # Note: find_package(omptest) may require setting of PATH_SUFFIXES
  #       Example: "lib/cmake/openmp/omptest", due to the install location
  install(TARGETS omptest
          EXPORT OPENMPomptest
          LIBRARY COMPONENT omptest
          DESTINATION "${OPENMP_INSTALL_LIBDIR}"
          INCLUDES DESTINATION "${OMPTEST_HEADER_INSTALL_DIR}")

  # Allow to link omptest by using: target_link_libraries( ... omptest::omptest)
  # Additionally, it automatically propagates the include directory.
  install(EXPORT OPENMPomptest
          DESTINATION "${OMPTEST_CONFIG_INSTALL_DIR}"
          NAMESPACE omptest::
          FILE omptest-targets.cmake)
endif()

# Discover unit tests (added to check-openmp)
if(LIBOMPTEST_BUILD_UNITTESTS)
  add_subdirectory(test)
endif()
