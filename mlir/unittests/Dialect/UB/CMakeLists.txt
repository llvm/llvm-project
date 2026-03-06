add_mlir_unittest(MLIRUBTests
  UBMatchersTest.cpp
)
mlir_target_link_libraries(MLIRUBTests
  PRIVATE
  MLIRArithDialect
  MLIRUBDialect
)
