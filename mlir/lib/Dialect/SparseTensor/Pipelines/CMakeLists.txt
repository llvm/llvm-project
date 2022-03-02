add_mlir_dialect_library(MLIRSparseTensorPipelines
  SparseTensorPipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SparseTensor

  LINK_LIBS PUBLIC
  MLIRArithmeticTransforms
  MLIRAffineToStandard
  MLIRBufferizationTransforms
  MLIRFuncTransforms
  MLIRLinalgTransforms
  MLIRMathToLLVM
  MLIRMemRefToLLVM
  MLIRPass
  MLIRReconcileUnrealizedCasts
  MLIRSCFToControlFlow
  MLIRSparseTensor
  MLIRSparseTensorTransforms
  MLIRTensorTransforms
  MLIRVectorToLLVM
  MLIRVectorTransforms
)
