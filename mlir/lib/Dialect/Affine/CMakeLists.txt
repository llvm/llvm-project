add_mlir_dialect_library(MLIRAffine
  IR/AffineOps.cpp
  IR/AffineValueMap.cpp
  EDSC/Builders.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Affine

  DEPENDS
  MLIRAffineOpsIncGen
  )
target_link_libraries(MLIRAffine
  PUBLIC
  MLIREDSC
  MLIRIR
  MLIRLoopLikeInterface
  MLIRSideEffects
  MLIRStandardOps
  )

add_subdirectory(Transforms)
add_subdirectory(Utils)
