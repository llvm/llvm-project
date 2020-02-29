add_mlir_conversion_library(MLIRLinalgToLLVM
  LinalgToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/LinalgToLLVM

  LINK_LIBS
  MLIRAffineToStandard
  MLIREDSC
  MLIRIR
  MLIRLinalgOps
  MLIRLLVMIR
  MLIRLoopToStandard
  MLIRPass
  MLIRStandardOps
  MLIRStandardToLLVM
  MLIRVectorToLLVM
  MLIRTransforms
  LLVMCore
  LLVMSupport
  )
