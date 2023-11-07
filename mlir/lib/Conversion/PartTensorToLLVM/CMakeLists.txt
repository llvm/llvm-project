add_mlir_conversion_library(MLIRPartTensorToLLVM
  PartTensorToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/PartTensorToLLVM

  DEPENDS
  MLIRConversionPassIncGen
  intrinsics_gen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRAffineToStandard
  MLIRIR
  MLIRPartTensor
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRMemRefToLLVM
  MLIRSCFToControlFlow
  MLIRTransforms
  MLIRVectorToLLVM
  MLIRVectorToSCF
  )
