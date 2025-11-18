add_mlir_dialect_library(MLIRGPUPipelines
  GPUToNVVMPipeline.cpp
  GPUToXeVMPipeline.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/GPU

  LINK_LIBS PUBLIC
  MLIRMemRefTransforms
  MLIRFuncDialect
  MLIRPass
  MLIRTransforms
  MLIRLinalgTransforms
  MLIRAffineToStandard
  MLIRGPUToLLVMSPV
  MLIRGPUToNVVMTransforms
  MLIRIndexToLLVM
  MLIRMathToLLVM
  MLIRMathToXeVM
  MLIRNVGPUToNVVM
  MLIRNVVMToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRSCFToControlFlow
  MLIRVectorToSCF
  MLIRXeGPUTransforms
  MLIRXeGPUToXeVM
  MLIRXeVMToLLVM
)
