add_lit_testsuite(check-llvm-filecheck "Running lit suite for FileCheck"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${LLVM_TEST_DEPENDS_COMMON}
  )
