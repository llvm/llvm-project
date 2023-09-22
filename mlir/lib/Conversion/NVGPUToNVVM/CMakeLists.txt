add_mlir_conversion_library(MLIRNVGPUToNVVM
  NVGPUToNVVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/NVGPUToNVVM

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRGPUDialect
  MLIRGPUToGPURuntimeTransforms
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRNVGPUDialect
  MLIRNVVMDialect
  MLIRPass
  MLIRSCFTransforms
  MLIRTransforms
  )
