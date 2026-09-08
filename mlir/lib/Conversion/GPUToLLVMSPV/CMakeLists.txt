add_mlir_conversion_library(MLIRGPUToLLVMSPV
  GPUToLLVMSPV.cpp

  LINK_LIBS PUBLIC
  MLIRGPUDialect
  MLIRGPUToGPURuntimeTransforms
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRSPIRVAttrToLLVMConversion
)
