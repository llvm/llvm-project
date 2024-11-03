add_mlir_conversion_library(MLIRMemRefToLLVM
  AllocLikeConversion.cpp
  MemRefToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/MemRefToLLVM

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRDataLayoutInterfaces
  MLIRLLVMCommonConversion
  MLIRMemRefDialect
  MLIRMemRefUtils
  MLIRLLVMDialect
  MLIRTransforms
  )
