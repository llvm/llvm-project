add_mlir_dialect_library(MLIRCIRTransforms
  LifetimeCheck.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/CIR

  DEPENDS
  MLIRCIRPassIncGen

  LINK_LIBS PUBLIC

  MLIRAnalysis
  MLIRIR
  MLIRCIR
  MLIRCIRAnalysis
  MLIRPass
)
