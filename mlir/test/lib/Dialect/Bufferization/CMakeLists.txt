# Exclude tests from libMLIR.so
add_mlir_library(MLIRBufferizationTestPasses
  TestTensorCopyInsertion.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRSCFDialect
  MLIRIR
  MLIRPass
)
