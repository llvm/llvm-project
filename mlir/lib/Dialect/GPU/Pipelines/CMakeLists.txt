add_mlir_dialect_library(MLIRGPUPipelines
  GPUToNVVMPipeline.cpp
  GPUToROCDLPipeline.cpp
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
  MLIRAMDGPUToROCDL
  MLIRArithToLLVM
  MLIRFuncToLLVM
  MLIRGPUToLLVMSPV
  MLIRGPUToNVVMTransforms
  MLIRGPUToROCDLTransforms
  MLIRIndexToLLVM
  MLIRMathToLLVM
  MLIRMathToXeVM
  MLIRNVGPUToNVVM
  MLIRNVVMToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRSCFToControlFlow
  MLIRVectorToLLVMPass
  MLIRVectorToSCF
  MLIRXeGPUTransforms
  MLIRXeGPUToXeVM
  MLIRXeVMToLLVM
)
