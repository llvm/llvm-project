add_mlir_dialect_library(MLIRVectorTransforms
  BufferizableOpInterfaceImpl.cpp
  Bufferize.cpp
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
  MLIRAffine
  MLIRAffineAnalysis
  MLIRAffineUtils
  MLIRArithmetic
  MLIRBufferization
  MLIRBufferizationTransforms
  MLIRDialectUtils
  MLIRIR
  MLIRLinalg
  MLIRMemRef
  MLIRSCF
  MLIRTransforms
  MLIRVector
  MLIRVectorInterfaces
  MLIRVectorUtils
  )
