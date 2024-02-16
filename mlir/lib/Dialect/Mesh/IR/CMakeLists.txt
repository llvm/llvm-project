add_mlir_dialect_library(MLIRMeshDialect
  MeshOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Mesh

  DEPENDS
  MLIRMeshIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRSupport
  MLIRViewLikeInterface
)
