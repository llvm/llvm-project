# Subset of dependencies for TableGen lit test.
set(LLVM_TEST_TABLEGEN_DEPENDS
  llvm-tblgen
  )

add_lit_testsuite(check-llvm-tablegen "Running lit suite for TableGen"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${LLVM_TEST_DEPENDS_COMMON}
  DEPENDS ${LLVM_TEST_TABLEGEN_DEPENDS}
  )
