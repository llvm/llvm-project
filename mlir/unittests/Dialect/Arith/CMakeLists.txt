add_mlir_unittest(MLIRArithOpsTests
  ArithOpsFoldersTest.cpp
)
target_link_libraries(MLIRArithOpsTests
  PRIVATE
  MLIRArithDialect
)
