# Exclude tests from libMLIR.so
add_mlir_library(MLIRTensorTestPasses
  TestTensorTransforms.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRArithmeticDialect
  MLIRLinalgDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTensorTransforms
  MLIRTransforms
  )
