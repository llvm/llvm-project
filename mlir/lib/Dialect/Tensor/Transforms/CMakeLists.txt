add_mlir_dialect_library(MLIRTensorTransforms
  Bufferize.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Tensor/Transforms

  DEPENDS
  MLIRTensorTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRArithmetic
  MLIRIR
  MLIRMemRef
  MLIRPass
  MLIRSCF
  MLIRTensor
  MLIRTransforms
  )
