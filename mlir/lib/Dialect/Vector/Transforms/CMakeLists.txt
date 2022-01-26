add_mlir_dialect_library(MLIRVectorTransforms
  BufferizableOpInterfaceImpl.cpp
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

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRAffineAnalysis
  MLIRAffineUtils
  MLIRArithmetic
  MLIRBufferization
  MLIRDialectUtils
  MLIRIR
  MLIRLinalg
  MLIRMemRef
  MLIRSCF
  MLIRVector
  MLIRVectorInterfaces
  MLIRVectorUtils
  )
