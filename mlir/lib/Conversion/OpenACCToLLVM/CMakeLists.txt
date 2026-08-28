add_mlir_conversion_library(MLIROpenACCToLLVM
  ACCAtomicPatterns.cpp
  ACCExecutableDirectivePatterns.cpp
  ACCToLLVMUtils.cpp
  ACCToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/OpenACCToLLVM

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithToLLVM
  MLIRComplexDialect
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRMemRefToLLVM
  MLIROpenACCAnalysis
  MLIROpenACCDialect
  MLIROpenACCUtils
  MLIRTransforms
)
