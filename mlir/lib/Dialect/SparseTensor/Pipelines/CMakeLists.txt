add_mlir_dialect_library(MLIRSparseTensorPipelines
  SparseTensorPipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SparseTensor

  # SparseTensorPipelines.cpp includes the aggregate Conversion/Passes.h
  # header, whose declarations include these generated dialect headers.
  HEADER_LIBS
  MLIRArmSMETransforms
  MLIROpenACCDialect
  MLIRTosaTransforms

  LINK_LIBS PUBLIC
  MLIRArithTransforms
  MLIRAffineToStandard
  MLIRBufferizationTransforms
  MLIRComplexToLLVM
  MLIRComplexToLibm
  MLIRComplexToStandard
  MLIRFuncTransforms
  MLIRGPUToNVVMTransforms
  MLIRGPUTransforms
  MLIRLinalgTransforms
  MLIRMathToLibm
  MLIRMathToLLVM
  MLIRMemRefToLLVM
  MLIRPass
  MLIRReconcileUnrealizedCasts
  MLIRSCFToControlFlow
  MLIRSparseTensorDialect
  MLIRSparseTensorTransforms
  MLIRTensorTransforms
  MLIRUBToLLVM
  MLIRVectorToLLVMPass
  MLIRVectorTransforms
)
