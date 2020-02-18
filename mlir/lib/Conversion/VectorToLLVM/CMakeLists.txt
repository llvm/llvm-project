add_mlir_conversion_library(MLIRVectorToLLVM
  ConvertVectorToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/VectorToLLVM

  LINK_LIBS
  MLIRIR
  MLIRLLVMIR
  MLIRPass
  MLIRStandardOps
  MLIRStandardToLLVM
  MLIRVectorOps
  MLIRTransforms
  MLIRTransformUtils
  LLVMCore
  LLVMSupport
)
