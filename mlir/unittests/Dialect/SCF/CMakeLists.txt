add_mlir_unittest(MLIRSCFTests
  LoopLikeSCFOpsTest.cpp
)
target_link_libraries(MLIRSCFTests
  PRIVATE
  MLIRIR
  MLIRSCFDialect
)
