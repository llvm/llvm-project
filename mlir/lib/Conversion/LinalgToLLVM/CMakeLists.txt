add_mlir_conversion_library(MLIRLinalgToLLVM
  LinalgToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/LinalgToLLVM

  DEPENDS
  MLIRConversionPassIncGen
  intrinsics_gen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRAffineToStandard
  MLIREDSC
  MLIRIR
  MLIRLinalgOps
  MLIRLLVMIR
  MLIRSCFToStandard
  MLIRStandardToLLVM
  MLIRTransforms
  MLIRVectorToLLVM
  MLIRVectorToSCF
  )
