# Exclude tests from libMLIR.so
add_mlir_library(MLIRControlFlowTestPasses
  TestAssert.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRControlFlowToLLVM
  MLIRFuncDialect
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRPass
  MLIRTransforms
)
