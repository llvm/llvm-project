add_lit_testsuite(check-llvm-unit "Running lit suite for LLVM unit tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  EXCLUDE_FROM_CHECK_ALL
  DEPENDS UnitTests
  )
