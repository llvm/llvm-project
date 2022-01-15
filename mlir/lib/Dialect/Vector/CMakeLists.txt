add_mlir_dialect_library(MLIRVector
  VectorDropLeadUnitDim.cpp
  VectorInsertExtractStridedSliceRewritePatterns.cpp
  VectorMultiDimReductionTransforms.cpp
  VectorOps.cpp
  VectorTransferOpTransforms.cpp
  VectorTransferSplitRewritePatterns.cpp
  VectorTransferPermutationMapRewritePatterns.cpp
  VectorTransforms.cpp
  VectorUnrollDistribute.cpp
  VectorUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Vector

  DEPENDS
  MLIRVectorOpsIncGen
  MLIRVectorOpsEnumsIncGen

  LINK_LIBS PUBLIC
  MLIRArithmetic
  MLIRDialectUtils
  MLIRIR
  MLIRStandard
  MLIRAffine
  MLIRAffineUtils
  MLIRLinalg
  MLIRMemRef
  MLIRSCF
  MLIRAffineAnalysis
  MLIRDataLayoutInterfaces
  MLIRSideEffectInterfaces
  MLIRVectorInterfaces
  )
