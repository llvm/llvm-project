# Exclude tests from libMLIR.so
add_mlir_library(MLIRTestTransforms
  TestCommutativityUtils.cpp
  TestConstantFold.cpp
  TestControlFlowSink.cpp
  TestInlining.cpp
  TestIntRangeInference.cpp

  EXCLUDE_FROM_LIBMLIR

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRInferIntRangeInterface
  MLIRTestDialect
  MLIRTransforms
  )

target_include_directories(MLIRTestTransforms
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../Dialect/Test
  ${CMAKE_CURRENT_BINARY_DIR}/../Dialect/Test
  )
