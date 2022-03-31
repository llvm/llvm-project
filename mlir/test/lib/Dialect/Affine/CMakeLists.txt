# Exclude tests from libMLIR.so
add_mlir_library(MLIRAffineTransformsTestPasses
  TestAffineDataCopy.cpp
  TestAffineLoopUnswitching.cpp
  TestAffineLoopParametricTiling.cpp
  TestLoopFusion.cpp
  TestLoopMapping.cpp
  TestLoopPermutation.cpp
  TestVectorizationUtils.cpp

  EXCLUDE_FROM_LIBMLIR

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Affine
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRAffineTransforms
  MLIRAffineUtils
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRVectorUtils
  )
