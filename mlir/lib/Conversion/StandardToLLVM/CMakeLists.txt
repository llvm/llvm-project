add_mlir_conversion_library(MLIRStandardToLLVM
  ConvertStandardToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/StandardToLLVM
)
target_link_libraries(
  MLIRStandardToLLVM
  PUBLIC
  MLIRLLVMIR
  MLIRTransforms
  LLVMCore
  LLVMSupport
  )
