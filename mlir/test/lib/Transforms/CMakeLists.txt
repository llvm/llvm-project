add_mlir_pdll_library(MLIRTestDialectConversionPDLLPatternsIncGen
  TestDialectConversion.pdll
  TestDialectConversionPDLLPatterns.h.inc

  EXTRA_INCLUDES
  ${CMAKE_CURRENT_SOURCE_DIR}/../Dialect/Test
  ${CMAKE_CURRENT_BINARY_DIR}/../Dialect/Test
  )

# Exclude tests from libMLIR.so
add_mlir_library(MLIRTestTransforms
  TestCommutativityUtils.cpp
  TestConstantFold.cpp
  TestControlFlowSink.cpp
  TestDialectConversion.cpp
  TestInlining.cpp
  TestIntRangeInference.cpp
  TestMakeIsolatedFromAbove.cpp
  TestTopologicalSort.cpp

  EXCLUDE_FROM_LIBMLIR

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms

  DEPENDS
  MLIRTestDialectConversionPDLLPatternsIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRFuncDialect
  MLIRInferIntRangeInterface
  MLIRTestDialect
  MLIRTransforms
  )

target_include_directories(MLIRTestTransforms
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../Dialect/Test
  ${CMAKE_CURRENT_BINARY_DIR}/../Dialect/Test
  )
