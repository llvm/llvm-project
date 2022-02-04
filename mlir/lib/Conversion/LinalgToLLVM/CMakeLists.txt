add_mlir_conversion_library(MLIRLinalgToLLVM
  LinalgToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/LinalgToLLVM

  DEPENDS
  MLIRConversionPassIncGen
  MLIRLinalgPassIncGen
  intrinsics_gen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRAffineToStandard
  MLIRIR
  MLIRLinalg
  MLIRLLVMCommonConversion
  MLIRLLVMIR
  MLIRMemRefToLLVM
  MLIRSCFToControlFlow
  MLIRTransforms
  MLIRVectorToLLVM
  MLIRVectorToSCF
  )
