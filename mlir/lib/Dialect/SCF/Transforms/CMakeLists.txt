add_mlir_dialect_library(MLIRSCFTransforms
  ParallelLoopFusion.cpp
  ParallelLoopSpecialization.cpp
  ParallelLoopTiling.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SCF

  DEPENDS
  MLIRSCFPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffineOps
  MLIRIR
  MLIRPass
  MLIRSCF
  MLIRStandardOps
  MLIRSupport
  )
