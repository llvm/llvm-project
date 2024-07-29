add_mlir_conversion_library(MLIRGPUToLLVMSPV
  GPUToLLVMSPV.cpp

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRGPUDialect
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRSPIRVDialect
)
