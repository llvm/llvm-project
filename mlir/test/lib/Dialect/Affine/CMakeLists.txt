# Exclude tests from libMLIR.so
add_mlir_library(MLIRAffineTransformsTestPasses
  TestAffineDataCopy.cpp
  TestAffineLoopUnswitching.cpp
  TestAffineLoopParametricTiling.cpp
  TestAccessAnalysis.cpp
  TestDecomposeAffineOps.cpp
  TestReifyValueBounds.cpp
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
  MLIRTestDialect
  )
mlir_target_link_libraries(MLIRAffineTransformsTestPasses PUBLIC
  MLIRArithTransforms
  MLIRAffineAnalysis
  MLIRAffineTransforms
  MLIRAffineUtils
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRMemRefDialect
  MLIRTensorDialect
  MLIRVectorUtils
  )

target_include_directories(MLIRAffineTransformsTestPasses
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../Test
  ${CMAKE_CURRENT_BINARY_DIR}/../Test
  )


# Static vectorization test utility conflicts.
set_source_files_properties(
  TestVectorizationUtils.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)

# Static value bound analysis test infrastructure conflicts.
set_source_files_properties(
  TestReifyValueBounds.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)

# Static analysis test harness redefinitions.
set_source_files_properties(
  TestAccessAnalysis.cpp
  TestAffineLoopUnswitching.cpp
  TestDecomposeAffineOps.cpp
  TestLoopFusion.cpp
  TestLoopPermutation.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
