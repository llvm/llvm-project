add_lit_testsuite(check-llvm-tablegen "Running lit suite for TableGen"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${LLVM_TEST_DEPENDS_COMMON}
  DEPENDS llvm-tblgen
  )
