add_mlir_conversion_library(MLIRStandardToLLVM
  StandardToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/StandardToLLVM

  DEPENDS
  MLIRConversionPassIncGen
  intrinsics_gen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRArithmeticToLLVM
  MLIRDataLayoutInterfaces
  MLIRLLVMCommonConversion
  MLIRLLVMIR
  MLIRMath
  MLIRTransforms
  )
