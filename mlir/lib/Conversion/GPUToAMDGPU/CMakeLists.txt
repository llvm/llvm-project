add_mlir_conversion_library(MLIRGPUToAMDGPUTransforms
  LowerGPUOpsToAMDGPUOps.cpp
  WmmaOpsToAMDGPU.cpp

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithToLLVM
  MLIRFuncToLLVM
  MLIRGPUDialect
  MLIRGPUToGPURuntimeTransforms
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRMemRefToLLVM
  MLIRROCDLDialect
  MLIRPass
  )
