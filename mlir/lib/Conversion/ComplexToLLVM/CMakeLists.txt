add_mlir_conversion_library(MLIRComplexToLLVM
  ComplexToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/ComplexToLLVM

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithAttrToLLVMConversion
  MLIRComplexDialect
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRTransforms
  )
