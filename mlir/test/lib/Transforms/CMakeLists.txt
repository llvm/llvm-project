set(LLVM_OPTIONAL_SOURCES 
  TestDialectConversion.cpp)
set(MLIRTestTransformsPDLDep)
set(MLIRTestTransformsPDLSrc)
if(MLIR_ENABLE_PDL_IN_PATTERNMATCH)
add_mlir_pdll_library(MLIRTestDialectConversionPDLLPatternsIncGen
  TestDialectConversion.pdll
  TestDialectConversionPDLLPatterns.h.inc

  EXTRA_INCLUDES
  ${CMAKE_CURRENT_SOURCE_DIR}/../Dialect/Test
  ${CMAKE_CURRENT_BINARY_DIR}/../Dialect/Test
  )
  set(MLIRTestTransformsPDLSrc
    TestDialectConversion.cpp)
  set(MLIRTestTransformsPDLDep
    MLIRTestDialectConversionPDLLPatternsIncGen)
endif()

# Exclude tests from libMLIR.so
add_mlir_library(MLIRTestTransforms
  TestCommutativityUtils.cpp
  TestConstantFold.cpp
  TestControlFlowSink.cpp
  TestInlining.cpp
  TestIntRangeInference.cpp
  TestMakeIsolatedFromAbove.cpp
  TestTopologicalSort.cpp
  ${MLIRTestTransformsPDLSrc}

  EXCLUDE_FROM_LIBMLIR

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms

  DEPENDS
  ${MLIRTestTransformsPDLDep}

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
