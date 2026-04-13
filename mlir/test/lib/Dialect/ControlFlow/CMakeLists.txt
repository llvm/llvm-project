# Exclude tests from libMLIR.so
add_mlir_library(MLIRControlFlowTestPasses
  TestAssert.cpp

  EXCLUDE_FROM_LIBMLIR
)
mlir_target_link_libraries(MLIRControlFlowTestPasses PUBLIC
  MLIRControlFlowToLLVM
  MLIRFuncDialect
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRPass
  MLIRTransforms
)
