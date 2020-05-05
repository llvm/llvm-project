add_mlir_conversion_library(MLIRVectorToLLVM
  ConvertVectorToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/VectorToLLVM

  DEPENDS
  MLIRConversionPassIncGen
  intrinsics_gen
  )

target_link_libraries(MLIRVectorToLLVM
  PUBLIC
  MLIRLLVMIR
  MLIRStandardToLLVM
  MLIRVector
  MLIRTransforms
  LLVMCore
  LLVMSupport
  )
