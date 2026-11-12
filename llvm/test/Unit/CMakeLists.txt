if (LLVM_TEST_WINDOWS_PREFER_FORWARD_SLASH_MATRIX)
  set(LLVM_UNIT_TEST_PATHS
    ${CMAKE_CURRENT_BINARY_DIR}/default
    ${CMAKE_CURRENT_BINARY_DIR}/forward_slash
    )
else()
  set(LLVM_UNIT_TEST_PATHS
    ${CMAKE_CURRENT_BINARY_DIR}
    )
endif()

add_lit_testsuite(check-llvm-unit "Running lit suite for LLVM unit tests"
  ${LLVM_UNIT_TEST_PATHS}
  EXCLUDE_FROM_CHECK_ALL
  DEPENDS UnitTests
  )
