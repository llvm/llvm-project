##===----------------------------------------------------------------------===##
#
# Build actual testing library.
#
##===----------------------------------------------------------------------===##

if(NOT ${LIBOMPTARGET_OMPT_SUPPORT})
  return()
endif()

cmake_minimum_required(VERSION 3.22)
project(omptest LANGUAGES CXX)

# Use LLVM-bundled GoogleTest
include(GoogleTest)

set(OMPTEST_HEADERS
  ./include/AssertMacros.h
  ./include/InternalEvent.h
  ./include/InternalEventCommon.h
  ./include/OmptAliases.h
  ./include/OmptAsserter.h
  ./include/OmptAssertEvent.h
  ./include/OmptCallbackHandler.h
  ./include/OmptTester.h
  ./include/OmptTesterGoogleTest.h
  ./include/OmptTesterStandalone.h
)

add_library(omptest
  SHARED

  ${OMPTEST_HEADERS}
  ./src/InternalEvent.cpp
  ./src/InternalEventOperators.cpp
  ./src/OmptAsserter.cpp
  ./src/OmptAssertEvent.cpp
  ./src/OmptCallbackHandler.cpp
  ./src/OmptTester.cpp
)

target_include_directories(omptest
  PRIVATE
  ${LIBOMPTARGET_INCLUDE_DIR}
)

# Include local include directory, so we do not need relative paths
include_directories(./include)
target_compile_features(omptest PRIVATE cxx_std_17)

# Link against gtest and gtest_main (disabled for now)
# TODO: Look into the "real" effects of PUBLIC in this scenario
#       Tests still need to link to gtest / gtest_main (otherwise: compile fail)
#       Those projects should not need to do this with PUBLIC or INTERFACE
# target_link_libraries(omptest INTERFACE gtest gtest_main)

# Create and install package configuration files.
configure_file(
  ${omptest_SOURCE_DIR}/cmake/omptest-config.cmake.in
  ${omptest_BINARY_DIR}/cmake/omptest-config.cmake @ONLY)

install(FILES ${omptest_BINARY_DIR}/cmake/omptest-config.cmake
        DESTINATION "${OFFLOAD_INSTALL_LIBDIR}/cmake/omptest")

# Install libomptest header files: Copy the whole include dir
install(DIRECTORY ./include
        DESTINATION "${OFFLOAD_INSTALL_LIBDIR}/omptest"
        FILES_MATCHING PATTERN "*.h")

install(TARGETS omptest LIBRARY COMPONENT omptest DESTINATION "${OFFLOAD_INSTALL_LIBDIR}")
