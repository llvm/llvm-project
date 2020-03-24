add_mlir_dialect_library(MLIRAffineTransforms
  AffineDataCopyGeneration.cpp
  AffineLoopInvariantCodeMotion.cpp
  LoopTiling.cpp
  LoopUnroll.cpp
  LoopUnrollAndJam.cpp
  SuperVectorize.cpp
  SimplifyAffineStructures.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Affine

  DEPENDS
  MLIRAffineOpsIncGen
  MLIRLoopLikeInterfaceIncGen
  )
target_link_libraries(MLIRAffineTransforms
  PUBLIC
  MLIRAffine
  MLIREDSC
  MLIRIR
  MLIRPass
  MLIRSideEffects
  MLIRStandardOps
  MLIRTransformUtils
  MLIRVector
  MLIRVectorToLLVM
  )

