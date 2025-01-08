# Exclude tests from libMLIR.so
add_mlir_library(MLIRTestMathToVCIX
  TestMathToVCIXConversion.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMathDialect
  MLIRVCIXDialect
  MLIRVectorDialect
  MLIRPass
  MLIRTransforms
)
