add_mlir_conversion_library(MLIRMemRefToLLVM
  MemRefToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/MemRefToLLVM

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRDataLayoutInterfaces
  MLIRFuncDialect
  MLIRLLVMCommonConversion
  MLIRMemRefDialect
  MLIRMemRefUtils
  MLIRLLVMDialect
  MLIRTransforms
  )
