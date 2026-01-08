add_mlir_conversion_library(MLIRAMDGPUToROCDL
  AMDGPUToROCDL.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/AMDGPUToROCDL

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  AMDGPUUtils
  Core

  LINK_LIBS PUBLIC
  MLIRAMDGPUDialect
  MLIRAMDGPUUtils
  MLIRGPUToGPURuntimeTransforms
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRPass
  MLIRROCDLDialect
  MLIRTransforms
  )
