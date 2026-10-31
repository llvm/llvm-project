set(LLVM_OPTIONAL_SOURCES
  BufferDeallocationOpInterfaceImpl.cpp
  BufferizableOpInterfaceImpl.cpp
  BufferViewFlowOpInterfaceImpl.cpp
  )

add_mlir_dialect_library(MLIRArithTransforms
  EmulateUnsupportedFloats.cpp
  EmulateWideInt.cpp
  EmulateNarrowType.cpp
  ExpandOps.cpp
  IntRangeOptimizations.cpp
  ReifyValueBounds.cpp
  ShardingInterfaceImpl.cpp
  UnsignedWhenEquivalent.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Arith/Transforms

  DEPENDS
  MLIRArithTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysisDataFlow
  MLIRArithCommonConversion
  MLIRArithDialect
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRInferIntRangeInterface
  MLIRIR
  MLIRLoopLikeInterface
  MLIRMemRefDialect
  MLIRShardDialect
  MLIRPass
  MLIRShardingInterface
  MLIRTensorDialect
  MLIRTransforms
  MLIRTransformUtils
  MLIRValueBoundsOpInterface
  MLIRVectorDialect
  )

add_mlir_dialect_library(MLIRArithBufferizableOpInterfaceImpl
  BufferizableOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Arith/Transforms

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRIR
  MLIRMemRefDialect
  )

add_mlir_dialect_library(MLIRArithBufferDeallocationOpInterfaceImpl
  BufferDeallocationOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Arith/Transforms

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRIR
  MLIRMemRefDialect
  )

add_mlir_dialect_library(MLIRArithBufferViewFlowOpInterfaceImpl
  BufferViewFlowOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Arith/Transforms

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRIR
  )
