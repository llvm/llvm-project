add_mlir_conversion_library(MLIRMPIToLLVM
  MPIToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/MPIToLLVM

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRMPIDialect
  )
