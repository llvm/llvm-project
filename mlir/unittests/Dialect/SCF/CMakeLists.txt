add_mlir_unittest(MLIRSCFTests
  LoopLikeSCFOpsTest.cpp
)
target_include_directories(MLIRSCFTests PRIVATE "${MLIR_BINARY_DIR}/test/lib/Dialect/Test")
mlir_target_link_libraries(MLIRSCFTests
  PRIVATE
  MLIRIR
  MLIRSCFDialect
  MLIRSCFUtils
  MLIRTensorDialect
)
target_link_libraries(MLIRSCFTests PRIVATE MLIRTestDialect)
