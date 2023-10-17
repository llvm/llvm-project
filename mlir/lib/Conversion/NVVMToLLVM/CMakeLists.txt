add_mlir_conversion_library(MLIRNVVMToLLVM
  NVVMToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/NVVMToLLVM

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRGPUDialect
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRNVVMDialect
  MLIRNVGPUDialect
  MLIRPass
  MLIRTransforms
  )
