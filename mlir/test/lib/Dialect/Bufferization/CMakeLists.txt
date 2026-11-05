# Exclude tests from libMLIR.so
add_mlir_library(MLIRBufferizationTestPasses
  TestOneShotModuleBufferize.cpp
  TestTensorCopyInsertion.cpp
  TestTensorLikeAndBufferLike.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRTestDialect
)
mlir_target_link_libraries(MLIRBufferizationTestPasses PUBLIC
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRIR
  MLIRPass
)
