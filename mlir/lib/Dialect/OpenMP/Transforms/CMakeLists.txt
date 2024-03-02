add_mlir_dialect_library(MLIROpenMPTransforms
  OpenMPTaskBasedTarget.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/OpenMP

  DEPENDS
  MLIROpenMPPassIncGen

  LINK_LIBS PUBLIC
  MLIROpenMPDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
)
