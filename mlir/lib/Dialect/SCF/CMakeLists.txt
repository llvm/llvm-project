file(GLOB globbed *.c *.cpp)
add_mlir_dialect_library(MLIRSCF
  ${globbed}
  EDSC/Builders.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/LoopOps

  DEPENDS
  MLIRSCFOpsIncGen

  LINK_LIBS PUBLIC
  MLIREDSC
  MLIRIR
  MLIRLoopLikeInterface
  MLIRSideEffects
  MLIRStandardOps
  )

add_subdirectory(Transforms)
