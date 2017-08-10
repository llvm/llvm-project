
if(APPLE)
  # LeakSanitizer is not supported on OSX and Windows right now
  set(HAS_LSAN 0)
  message(WARNING "LeakSanitizer is not supported."
    " Building and running LibFuzzer LeakSanitizer tests is disabled."
    )
else()
  set(HAS_LSAN 1)
endif()

###############################################################################
# Unit tests
###############################################################################

add_custom_target(FuzzerUnitTests)
set_target_properties(FuzzerUnitTests PROPERTIES FOLDER "libFuzzer tests")

add_executable(LLVMFuzzer-Unittest FuzzerUnittest.cpp)

target_link_libraries(LLVMFuzzer-Unittest
  gtest
  gtest_main
  LLVMFuzzerNoMain
  )

target_include_directories(LLVMFuzzer-Unittest PRIVATE
  "${LLVM_MAIN_SRC_DIR}/utils/unittest/googletest/include"
  )

set_target_properties(LLVMFuzzer-Unittest
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY
  "${CMAKE_CURRENT_BINARY_DIR}"
)

include_directories(..)

###############################################################################
# Configure lit to run the tests
#
# Note this is done after declaring all tests so we can inform lit if any tests
# need to be disabled.
###############################################################################

# Use just-built Clang to compile/link tests on all platforms, except for
# Windows where we need to use clang-cl instead.
set(LIBFUZZER_TEST_COMPILER ${LLVM_RUNTIME_OUTPUT_INTDIR}/clang)
set(LIBFUZZER_TEST_CXX_COMPILER ${LLVM_RUNTIME_OUTPUT_INTDIR}/clang++)

# LIT-based libFuzzer tests.
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  )

# libFuzzer unit tests.
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/unit/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/unit/lit.site.cfg
  )

add_lit_testsuite(check-fuzzer "Running Fuzzer tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS LLVMFuzzer-Unittest)

add_dependencies(check-fuzzer LLVMFuzzer asan clang llvm-symbolizer FileCheck sancov not)
