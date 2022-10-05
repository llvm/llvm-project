add_mlir_dialect_library(MLIRGPUTransformOps
  GPUTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/GPU/TransformOps

  DEPENDS
  MLIRGPUTransformOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRGPUTransforms
  MLIRParser
  MLIRPDLDialect
  MLIRSideEffectInterfaces
  MLIRTransformDialect
  MLIRGPUOps
  )  
