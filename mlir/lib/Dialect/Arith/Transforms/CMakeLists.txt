add_mlir_pdll_library(MLIRArithPDLLPatternsIncGen
  TestReduceFloatBitwidth.pdll
  TestReduceFloatBitwidthPatterns.h.inc

  EXTRA_INCLUDES
  ${CMAKE_CURRENT_SOURCE_DIR}/../../Dialect/Test
  ${CMAKE_CURRENT_BINARY_DIR}/../../Dialect/Test
  )

add_mlir_dialect_library(MLIRArithTransforms
  BufferDeallocationOpInterfaceImpl.cpp
  BufferizableOpInterfaceImpl.cpp
  BufferViewFlowOpInterfaceImpl.cpp
  EmulateUnsupportedFloats.cpp
  EmulateWideInt.cpp
  EmulateNarrowType.cpp
  ExpandOps.cpp
  IntRangeOptimizations.cpp
  ReifyValueBounds.cpp
  TestReduceFloatBitwidth.cpp
  TestReduceFloatBitwidthConversion.cpp
  UnsignedWhenEquivalent.cpp

  ADDITIONAL_HEADER_DIRS
  {$MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Arith/Transforms

  DEPENDS
  MLIRArithPDLLPatternsIncGen
  MLIRArithTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRInferIntRangeInterface
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRPDLInterpDialect
  MLIRPDLDialect
  MLIRSupport
  MLIRTensorDialect
  MLIRTransforms
  MLIRTransformUtils
  MLIRValueBoundsOpInterface
  MLIRVectorDialect
  )
