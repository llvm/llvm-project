add_mlir_conversion_library(MLIRVectorToLoops
  ConvertVectorToLoops.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/VectorToLoops
)

target_link_libraries(MLIRVectorToLoops
  PUBLIC
  MLIREDSC
  MLIRAffineEDSC
  MLIRLLVMIR
  MLIRTransforms
  LLVMCore
  LLVMSupport
  )
