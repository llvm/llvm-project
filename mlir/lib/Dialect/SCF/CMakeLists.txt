add_mlir_dialect_library(MLIRSCF
  SCF.cpp
  EDSC/Builders.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/LoopOps

  DEPENDS
  MLIRSCFOpsIncGen

  LINK_LIBS PUBLIC
  MLIREDSC
  MLIRIR
  MLIRLoopLikeInterface
  MLIRSideEffectInterfaces
  MLIRStandard
  )

add_subdirectory(Transforms)
