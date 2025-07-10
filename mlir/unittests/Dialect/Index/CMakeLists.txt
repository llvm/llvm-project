add_mlir_unittest(MLIRIndexOpsTests
  IndexOpsFoldersTest.cpp
)
mlir_target_link_libraries(MLIRIndexOpsTests
  PRIVATE
  MLIRIndexDialect
)
