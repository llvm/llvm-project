add_mlir_dialect_library(MLIRXeGPUTransformOps
  XeGPUTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/mlir/Dialect/XeGPU/TransformOps/

  DEPENDS
  MLIRXeGPUTransformOpsIncGen

  LINK_LIBS PUBLIC
  MLIRXeGPUDialect
  MLIRXeGPUTransforms
  MLIRIR
  MLIRTransformDialect
  MLIRFuncDialect
  MLIRSCFDialect
)
