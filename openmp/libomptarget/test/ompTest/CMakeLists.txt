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

set(OMPTEST_HEADERS
  ./include/AssertMacros.h
  ./include/InternalEvent.h
  ./include/InternalEventCommon.h
  ./include/OmptAliases.h
  ./include/OmptAsserter.h
  ./include/OmptAssertEvent.h
  ./include/OmptCallbackHandler.h
  ./include/OmptTester.h
)

add_llvm_library(omptest
  SHARED

  ${OMPTEST_HEADERS}
  ./src/InternalEvent.cpp
  ./src/InternalEventOperators.cpp
  ./src/OmptAsserter.cpp
  ./src/OmptAssertEvent.cpp
  ./src/OmptCallbackHandler.cpp
  ./src/OmptTester.cpp

  ADDITIONAL_HEADER_DIRS
  ${LIBOMPTARGET_INCLUDE_DIR}
  ${LIBOMPTARGET_DEP_LIBFFI_INCLUDE_DIR}

  LINK_LIBS
  PRIVATE
  MemoryManager
  ${LIBOMPTARGET_DEP_LIBFFI_LIBRARIES}
  ${OPENMP_PTHREAD_LIB}

  NO_INSTALL_RPATH
)

target_compile_features(omptest PRIVATE cxx_std_17)

# Create and install package configuration files.
configure_file(
  ${omptest_SOURCE_DIR}/cmake/omptest-config.cmake.in
  ${omptest_BINARY_DIR}/cmake/omptest-config.cmake @ONLY)

install(FILES ${omptest_BINARY_DIR}/cmake/omptest-config.cmake
        DESTINATION lib/cmake/omptest)

# Install libomptest header files: Copy the whole include dir
install(DIRECTORY ./include
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/omptest
        FILES_MATCHING PATTERN "*.h")
