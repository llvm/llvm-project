add_mlir_dialect_library(MLIRSCF
  SCF.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/LoopOps

  DEPENDS
  MLIRSCFOpsIncGen

  LINK_LIBS PUBLIC
  MLIRArithmetic
  MLIRBufferization
  MLIRIR
  MLIRLoopLikeInterface
  MLIRSideEffectInterfaces
  MLIRStandard
  )

add_subdirectory(Transforms)
add_subdirectory(Utils)
