add_mlir_conversion_library(MLIRStandardToLLVM
  ConvertStandardToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/StandardToLLVM

  LINK_LIBS
  MLIRIR
  MLIRLLVMIR
  MLIRPass
  MLIRStandardOps
  MLIRTransforms
  LLVMCore
  LLVMSupport
  )
add_dependencies(
  MLIRStandardToLLVM

  MLIRLLVMIR
  MLIRTransforms
  LLVMCore
  LLVMSupport
)
