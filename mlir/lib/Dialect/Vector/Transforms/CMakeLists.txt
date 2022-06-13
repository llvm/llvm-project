add_mlir_dialect_library(MLIRVectorTransforms
  BufferizableOpInterfaceImpl.cpp
  Bufferize.cpp
  VectorDistribute.cpp
  VectorDropLeadUnitDim.cpp
  VectorInsertExtractStridedSliceRewritePatterns.cpp
  VectorMultiDimReductionTransforms.cpp
  VectorTransferOpTransforms.cpp
  VectorTransferSplitRewritePatterns.cpp
  VectorTransferPermutationMapRewritePatterns.cpp
  VectorTransforms.cpp
  VectorUnrollDistribute.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Vector/Transforms

  DEPENDS
  MLIRVectorTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAffineAnalysis
  MLIRAffineUtils
  MLIRArithmeticDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRDialectUtils
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTransforms
  MLIRVectorDialect
  MLIRVectorInterfaces
  MLIRVectorUtils
  )
