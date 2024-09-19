add_mlir_dialect_library(MLIRXeGPUTransforms
  XeGPUFoldAliasOps.cpp
  XeGPUDistributeToWI.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/XeGPU

  DEPENDS
  MLIRXeGPUPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffineUtils
  MLIRIR
  MLIRMemRefDialect
  MLIRXeGPUDialect
  MLIRVectorDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRPass
  MLIRTransforms
)
