add_mlir_conversion_library(MLIRArithToLLVM
  ArithToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/ArithToLLVM

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithAttrToLLVMConversion
  MLIRArithDialect
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  )
