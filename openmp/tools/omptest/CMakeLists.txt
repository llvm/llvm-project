##===----------------------------------------------------------------------===##
#
# Build OMPT unit testing library: ompTest
#
##===----------------------------------------------------------------------===##

cmake_minimum_required(VERSION 3.20)
project(omptest LANGUAGES CXX)

option(LIBOMPTEST_BUILD_STANDALONE
       "Build ompTest 'standalone', i.e. w/o GoogleTest."
       ${OPENMP_STANDALONE_BUILD})
option(LIBOMPTEST_BUILD_UNITTESTS
       "Build ompTest's unit tests, requires GoogleTest." OFF)
option(LIBOMPTEST_INSTALL_COMPONENTS
       "Install ompTest library, headers and package files." OFF)

# Exit early if OMPT support or LLVM-tests were disabled by the user.
if((NOT ${LIBOMP_OMPT_SUPPORT}) OR (NOT ${LLVM_INCLUDE_TESTS}))
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

  # Set and check GTest's source directory.
  set(OMPTEST_GTEST_PATH ${LLVM_THIRD_PARTY_DIR}/unittest/googletest)
  if(NOT EXISTS "${OMPTEST_GTEST_PATH}/src/gtest-all.cc")
    message(FATAL_ERROR "Missing gtest-all.cc under ${OMPTEST_GTEST_PATH}. "
                        "Make sure LLVM_THIRD_PARTY_DIR is set properly.")
  endif()

  # Build GTest as OBJECT library, so we can merge it into omptest.
  add_library(omptest_gtest OBJECT "${OMPTEST_GTEST_PATH}/src/gtest-all.cc")

  # Ensure PIC for inclusion into a shared library on ELF platforms.
  set_target_properties(omptest_gtest PROPERTIES POSITION_INDEPENDENT_CODE ON)

  # Set GTest include directories for omptest_gtest
  target_include_directories(omptest_gtest PRIVATE
    "${OMPTEST_GTEST_PATH}"
    "${OMPTEST_GTEST_PATH}/include"
  )

  # Set GTest include directories for omptest
  target_include_directories(omptest PUBLIC
    $<BUILD_INTERFACE:${OMPTEST_GTEST_PATH}>
    $<BUILD_INTERFACE:${OMPTEST_GTEST_PATH}/include>
  )

  # Avoid using LLVM-specific ostream helpers inside GoogleTest.
  target_compile_definitions(omptest_gtest PRIVATE GTEST_NO_LLVM_SUPPORT=1)

  # Add GoogleTest-based header and merge GTest into the shared lib.
  target_sources(omptest PRIVATE
    ./include/OmptTesterGoogleTest.h
    $<TARGET_OBJECTS:omptest_gtest>
  )

  # Link against LLVMSupport and Threads (recommended for GTest).
  find_package(Threads REQUIRED)
  target_link_libraries(omptest PUBLIC LLVMSupport Threads::Threads)

  # Ensure that embedded GTest symbols are exported from libomptest.so even in
  # builds that default to hidden.
  set_target_properties(omptest PROPERTIES
    CXX_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN OFF
  )

  # Make sure correct include directories are used, e.g. by the unit tests.
  # Otherwise, ABI-checks may fail.
  if(DEFINED LLVM_INCLUDE_DIRS)
    target_include_directories(omptest PUBLIC ${LLVM_INCLUDE_DIRS})
  endif()
else()
  # Add 'standalone' compile definitions
  target_compile_definitions(omptest PRIVATE
    -DOPENMP_LIBOMPTEST_BUILD_STANDALONE)

  # Add 'standalone' source files
  target_sources(omptest PRIVATE
    ./include/OmptTesterStandalone.h
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
  #       Example: "lib/cmake/openmp/omptest", this is due to the install location
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
