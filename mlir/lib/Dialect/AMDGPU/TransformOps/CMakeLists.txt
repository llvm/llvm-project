add_mlir_dialect_library(MLIRAMDGPUTransformOps
  AMDGPUTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/AMDGPU/TransformOps

  DEPENDS
  MLIRAMDGPUTransformOpsIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRArithDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRAMDGPUDialect
  MLIRAMDGPUTransforms
  MLIRParser
  MLIRSideEffectInterfaces
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTransformDialect
  MLIRTransformDialectUtils
  MLIRVectorTransforms

  )
