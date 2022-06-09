add_mlir_dialect_library(MLIRSCFTransformOps
  SCFTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SCF/TransformOps

  DEPENDS
  MLIRSCFTransformOpsIncGen

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRFunc
  MLIRIR
  MLIRPDL
  MLIRSCF
  MLIRSCFTransforms
  MLIRSCFUtils
  MLIRTransformDialect
  MLIRVector
)
