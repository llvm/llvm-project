add_mlir_conversion_library(MLIRGPUToNVVMTransforms
  LowerGpuOpsToNVVMOps.cpp
  WmmaOpsToNvvm.cpp

  LINK_LIBS PUBLIC
  MLIRArithToLLVM
  MLIRFuncToLLVM
  MLIRGPUDialect
  MLIRGPUToGPURuntimeTransforms
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRMathToNVVM
  MLIRMemRefToLLVM
  MLIRNVGPUDialect
  MLIRNVGPUToNVVM
  MLIRNVVMDialect
  MLIRPass
  MLIRTransformUtils
  )
