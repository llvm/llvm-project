add_mlir_dialect_library(MLIRVectorTransforms
  BufferizableOpInterfaceImpl.cpp
  Bufferize.cpp
  LowerVectorBroadcast.cpp
  LowerVectorContract.cpp
  LowerVectorGather.cpp
  LowerVectorMask.cpp
  LowerVectorMultiReduction.cpp
  LowerVectorScan.cpp
  LowerVectorShapeCast.cpp
  LowerVectorTransfer.cpp
  LowerVectorTranspose.cpp
  VectorDistribute.cpp
  VectorDropLeadUnitDim.cpp
  VectorEmulateNarrowType.cpp
  VectorInsertExtractStridedSliceRewritePatterns.cpp
  VectorTransferOpTransforms.cpp
  VectorTransferSplitRewritePatterns.cpp
  VectorTransforms.cpp
  VectorUnroll.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Vector/Transforms

  DEPENDS
  MLIRVectorTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAffineAnalysis
  MLIRAffineUtils
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRDialectUtils
  MLIRGPUDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRMemRefUtils
  MLIRSCFDialect
  MLIRSideEffectInterfaces
  MLIRTensorDialect
  MLIRTransforms
  MLIRVectorDialect
  MLIRVectorInterfaces
  MLIRVectorUtils
  )
