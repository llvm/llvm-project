add_mlir_conversion_library(MLIRVectorToLLVM
  ConvertVectorToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/VectorToLLVM

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core
  )

target_link_libraries(MLIRVectorToLLVM
  PUBLIC
  MLIRLLVMIR
  MLIRStandardToLLVM
  MLIRVector
  MLIRTransforms
  )
