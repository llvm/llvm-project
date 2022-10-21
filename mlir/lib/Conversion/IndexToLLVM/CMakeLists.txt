add_mlir_conversion_library(MLIRIndexToLLVM
  IndexToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/IndexToLLVM

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIndexDialect
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  )
