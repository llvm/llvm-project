# Exclude tests from libMLIR.so
add_mlir_library(MLIRBufferizationTestPasses
  TestTensorCopyInsertion.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRIR
  MLIRPass
)
