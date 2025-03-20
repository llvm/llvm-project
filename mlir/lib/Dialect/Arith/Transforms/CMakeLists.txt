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
  ShardingInterfaceImpl.cpp
  UnsignedWhenEquivalent.cpp

  ADDITIONAL_HEADER_DIRS
  {$MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Arith/Transforms

  DEPENDS
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
  MLIRMeshDialect
  MLIRPass
  MLIRShardingInterface
  MLIRTensorDialect
  MLIRTransforms
  MLIRTransformUtils
  MLIRValueBoundsOpInterface
  MLIRVectorDialect
  )
