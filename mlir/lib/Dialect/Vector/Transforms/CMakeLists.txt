add_mlir_dialect_library(MLIRVectorTransforms
  BufferizableOpInterfaceImpl.cpp
  Bufferize.cpp
  LowerVectorMask.cpp
  VectorDistribute.cpp
  VectorDropLeadUnitDim.cpp
  VectorInsertExtractStridedSliceRewritePatterns.cpp
  VectorMultiDimReductionTransforms.cpp
  VectorTransferOpTransforms.cpp
  VectorTransferSplitRewritePatterns.cpp
  VectorTransferPermutationMapRewritePatterns.cpp
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
  MLIRGPUOps
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSideEffectInterfaces
  MLIRTransforms
  MLIRVectorDialect
  MLIRVectorInterfaces
  MLIRVectorUtils
  )
