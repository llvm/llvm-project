add_mlir_dialect_library(MLIRLoopOpsTransforms
  ParallelLoopFusion.cpp
  ParallelLoopSpecialization.cpp
  ParallelLoopTiling.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/LoopOps

  DEPENDS
  MLIRLoopPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffineOps
  MLIRIR
  MLIRPass
  MLIRLoopOps
  MLIRStandardOps
  MLIRSupport
  )
