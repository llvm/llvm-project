add_mlir_dialect_library(MLIRGPUPipelines
  GPUToNVVMPipeline.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/GPU

  LINK_LIBS PUBLIC
  MLIRMemRefTransforms
  MLIRFuncDialect
  MLIRPass
  MLIRTransforms
  MLIRLinalgTransforms
  MLIRAffineToStandard
  MLIRGPUToNVVMTransforms
  MLIRIndexToLLVM
  MLIRMathToLLVM
  MLIRNVGPUToNVVM
  MLIRNVVMToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRSCFToControlFlow
  MLIRVectorToSCF
)
