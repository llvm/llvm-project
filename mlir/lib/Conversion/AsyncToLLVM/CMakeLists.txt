add_mlir_conversion_library(MLIRAsyncToLLVM
  AsyncToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/AsyncToLLVM

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithmetic
  MLIRAsync
  MLIRFuncTransforms
  MLIRLLVMCommonConversion
  MLIRLLVMIR
  MLIRStandardToLLVM
  MLIRTransforms
  )
