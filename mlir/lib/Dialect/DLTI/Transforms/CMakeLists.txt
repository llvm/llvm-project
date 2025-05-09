add_mlir_dialect_library(MLIRDLTITransforms
  SetTargetSpecs.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/DLTI

  DEPENDS
  MLIRDLTIPassIncGen

  LINK_LIBS PUBLIC
  MLIRDataLayoutInterfaces
  )
