# Exclude tests from libMLIR.so
add_mlir_library(MLIRTestMathToVCIX
  TestMathToVCIXConversion.cpp

  EXCLUDE_FROM_LIBMLIR
)
mlir_target_link_libraries(MLIRTestMathToVCIX PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMathDialect
  MLIRVCIXDialect
  MLIRVectorDialect
  MLIRPass
  MLIRTransforms
)
