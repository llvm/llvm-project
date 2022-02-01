add_mlir_dialect_library(MLIRVectorTransforms
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
  MLIRDialectUtils
  MLIRIR
  MLIRLinalg
  MLIRMemRef
  MLIRSCF
  MLIRStandard
  MLIRVector
  MLIRVectorInterfaces
  MLIRVectorUtils
  )
