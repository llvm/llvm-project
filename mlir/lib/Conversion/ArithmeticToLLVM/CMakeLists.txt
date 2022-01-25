add_mlir_conversion_library(MLIRArithmeticToLLVM
  ArithmeticToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/ArithmeticToLLVM

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithmetic
  MLIRLLVMCommonConversion
  MLIRLLVMIR
  )
