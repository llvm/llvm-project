add_mlir_conversion_library(MLIRLibCToLLVM
  LibCToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/LibCToLLVM

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRLibCDialect
  MLIRPass
  )
