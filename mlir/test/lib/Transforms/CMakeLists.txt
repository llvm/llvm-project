set(LLVM_TARGET_DEFINITIONS TestTransformsOps.td)
mlir_tablegen(TestTransformsOps.h.inc -gen-op-decls)
mlir_tablegen(TestTransformsOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRTestTransformsOpsIncGen)

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
  TestCompositePass.cpp
  TestConstantFold.cpp
  TestControlFlowSink.cpp
  TestInlining.cpp
  TestMakeIsolatedFromAbove.cpp
  TestTransformsOps.cpp
  ${MLIRTestTransformsPDLSrc}

  EXCLUDE_FROM_LIBMLIR

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms

  DEPENDS
  ${MLIRTestTransformsPDLDep}
  MLIRTestTransformsOpsIncGen

  LINK_LIBS PUBLIC
  MLIRTestDialect
  )
mlir_target_link_libraries(MLIRTestTransforms PUBLIC
  MLIRAnalysis
  MLIRFuncDialect
  MLIRInferIntRangeInterface
  MLIRTransforms
  MLIRTransformDialect
  )

target_include_directories(MLIRTestTransforms
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../Dialect/Test
  ${CMAKE_CURRENT_BINARY_DIR}/../Dialect/Test
  )
