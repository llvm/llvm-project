add_mlir_dialect_library(MLIRXeGPUTransforms
  XeGPUBlocking.cpp
  XeGPUFoldAliasOps.cpp
  XeGPUSubgroupDistribute.cpp
  XeGPUUnroll.cpp
  XeGPUWgToSgDistribute.cpp
  XeGPUPropagateLayout.cpp
  XeGPUVectorLinearize.cpp
  XeGPUPeepHoleOptimizer.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/XeGPU

  DEPENDS
  MLIRXeGPUPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffineUtils
  MLIRIR
  MLIRIndexDialect
  MLIRMemRefDialect
  MLIRXeGPUDialect
  MLIRPass
  MLIRTransforms
  MLIRGPUDialect
  MLIRXeGPUUtils
  MLIRGPUUtils
  MLIRVectorTransforms
)
