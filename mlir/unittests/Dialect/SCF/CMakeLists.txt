add_mlir_unittest(MLIRSCFTests
  LoopLikeSCFOpsTest.cpp
)
mlir_target_link_libraries(MLIRSCFTests
  PRIVATE
  MLIRIR
  MLIRSCFDialect
)
