# Exclude tests from libMLIR.so
add_mlir_library(MLIRBufferizationTestPasses
  TestTensorCopyInsertion.cpp

  EXCLUDE_FROM_LIBMLIR
)
mlir_target_link_libraries(MLIRBufferizationTestPasses PUBLIC
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRIR
  MLIRPass
)
