add_llvm_library(MLIRAffineTransformsTestPasses
  TestAffineDataCopy.cpp
  TestAffineLoopUnswitching.cpp
  TestLoopPermutation.cpp
  TestParallelismDetection.cpp
  TestVectorizationUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Affine
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR
  )

target_link_libraries(MLIRAffineTransformsTestPasses PRIVATE
  MLIRIR
  MLIRPass
  MLIRAffineTransforms
  MLIRSupport
  )
