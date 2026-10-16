add_mlir_conversion_library(MLIROpenMPToLLVM
  OpenMPToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/OpenMPToLLVM

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithToLLVM
  MLIRFuncToLLVM
  MLIRIRCore
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRMemRefToLLVM
  MLIROpenMPDialect

  HEADER_LIBS
  MLIRTransforms
  )
