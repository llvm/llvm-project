# Exclude tests from libMLIR.so
add_mlir_library(MLIRArithmeticTestPasses
  TestEmulateWideInt.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRArithmeticDialect
  MLIRArithmeticTransforms
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRPass
  MLIRVectorDialect
)
