##===----------------------------------------------------------------------===##
#
# Build OMPT unit testing library: ompTest
#
##===----------------------------------------------------------------------===##

cmake_minimum_required(VERSION 3.22)
project(omptest LANGUAGES CXX)

option(LIBOMPTEST_BUILD_STANDALONE
       "Build ompTest 'standalone', i.e. w/o GoogleTest."
       ${OPENMP_STANDALONE_BUILD})
option(LIBOMPTEST_BUILD_UNITTESTS
       "Build ompTest's unit tests, requires GoogleTest." OFF)

# In absence of corresponding OMPT support: exit early
if(NOT ${LIBOMP_OMPT_SUPPORT})
  return()
endif()

include(CMakePackageConfigHelpers)

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
# On (implicit) request of GoogleTest, link against the one provided with LLVM.
if ((NOT LIBOMPTEST_BUILD_STANDALONE) OR LIBOMPTEST_BUILD_UNITTESTS)
  # Check if standalone build was requested together with unittests
  if (LIBOMPTEST_BUILD_STANDALONE)
    # Emit warning: this build actually depends on LLVM's GoogleTest
    message(Warning "LIBOMPTEST_BUILD_STANDALONE and LIBOMPTEST_BUILD_UNITTESTS"
                    " requested simultaneously.\n"
                    "Linking against LLVM's GoogleTest library archives.\n"
                    "Disable LIBOMPTEST_BUILD_UNITTESTS to perform an actual"
                    " standalone build.")
    # Explicitly disable LIBOMPTEST_BUILD_STANDALONE
    set(LIBOMPTEST_BUILD_STANDALONE OFF)
  endif()

  # Use LLVM's gtest library archive
  set(GTEST_LIB "${LLVM_BINARY_DIR}/lib/libllvm_gtest.a")

  # Link gtest as whole-archive to expose required symbols
  set(GTEST_LINK_CMD "-Wl,--whole-archive" ${GTEST_LIB}
                     "-Wl,--no-whole-archive" LLVMSupport)

  # Add GoogleTest-based header
  target_sources(omptest PRIVATE ./include/OmptTesterGoogleTest.h)

  # Add LLVM-provided GoogleTest include directories.
  target_include_directories(omptest PRIVATE
    ${LLVM_THIRD_PARTY_DIR}/unittest/googletest/include)

  # TODO: Re-visit ABI breaking checks, disable for now.
  target_compile_definitions(omptest PUBLIC
    -DLLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING)

  # Link against gtest and gtest_main
  target_link_libraries(omptest PRIVATE ${GTEST_LINK_CMD})
else()
  # Add 'standalone' compile definitions
  target_compile_definitions(omptest PRIVATE
    -DOPENMP_LIBOMPTEST_BUILD_STANDALONE)

  # Add 'standalone' source files
  target_sources(omptest PRIVATE
    ./include/OmptTesterStandalone.h
    ./src/OmptTesterStandalone.cpp)
endif()

# Add common include directories.
target_include_directories(omptest PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${LIBOMP_HEADERS_INSTALL_PATH}/omptest>
)

target_compile_features(omptest PRIVATE cxx_std_17)

# Create and install package configuration files.
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/omptest-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/cmake/omptest-config.cmake
  INSTALL_DESTINATION "${OPENMP_INSTALL_LIBDIR}/cmake/openmp/omptest"
)

install(FILES ${omptest_BINARY_DIR}/cmake/omptest-config.cmake
        DESTINATION "${OPENMP_INSTALL_LIBDIR}/cmake/openmp/omptest")

# Install libomptest header files: Copy header-files from include dir
install(DIRECTORY ./include/
        DESTINATION "${LIBOMP_HEADERS_INSTALL_PATH}/omptest"
        FILES_MATCHING PATTERN "*.h")

# Install library and export targets.
# Note: find_package(omptest) may require setting of PATH_SUFFIXES
#       Example: "lib/cmake/openmp/omptest", this is due to the install location
install(TARGETS omptest
        EXPORT OPENMPomptest
        LIBRARY COMPONENT omptest
        DESTINATION "${OPENMP_INSTALL_LIBDIR}"
        INCLUDES DESTINATION "${LIBOMP_HEADERS_INSTALL_PATH}/omptest")

# Allow to link omptest by using: target_link_libraries( ... omptest::omptest)
# Additionally, it automatically propagates the include directory.
install(EXPORT OPENMPomptest
        DESTINATION "${OPENMP_INSTALL_LIBDIR}/cmake/openmp/omptest"
        NAMESPACE omptest::
        FILE omptest-targets.cmake)

# Discover unit tests (added to check-openmp)
if(LIBOMPTEST_BUILD_UNITTESTS)
  add_subdirectory(test)
endif()
