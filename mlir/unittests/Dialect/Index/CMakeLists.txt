add_mlir_unittest(MLIRIndexOpsTests
  IndexOpsFoldersTest.cpp
)
target_link_libraries(MLIRIndexOpsTests
  PRIVATE
  MLIRIndexDialect
)
