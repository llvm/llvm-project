add_mlir_conversion_library(MLIRStandardToLLVM
  StandardToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/StandardToLLVM

  DEPENDS
  MLIRConversionPassIncGen
)
target_link_libraries(
  MLIRStandardToLLVM
  PUBLIC
  MLIRLLVMIR
  MLIRTransforms
  LLVMCore
  LLVMSupport
  )
