add_mlir_conversion_library(MLIRGPUToAMDGPU
  GPUToAMDGPU.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/GPUToAMDGPU
  
  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRGPUDialect
  MLIRAMDGPUDialect
  MLIRAMDGPUUtils
  MLIRROCDLDialect
  MLIRPass
  MLIRTransforms
  )
