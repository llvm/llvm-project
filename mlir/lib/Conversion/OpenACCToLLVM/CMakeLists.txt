add_mlir_conversion_library(MLIROpenACCToLLVM
  ACCExecutableDirectivePatterns.cpp
  ACCToLLVMUtils.cpp
  ACCToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/OpenACCToLLVM

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithToLLVM
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIROpenACCDialect
  MLIROpenACCUtils
  MLIRTransforms
)
