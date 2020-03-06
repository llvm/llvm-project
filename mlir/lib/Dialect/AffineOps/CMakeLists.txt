add_mlir_dialect_library(MLIRAffineOps
  AffineOps.cpp
  AffineValueMap.cpp
  EDSC/Builders.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/AffineOps
  )
add_dependencies(MLIRAffineOps
  MLIRAffineOpsIncGen
  MLIRLoopLikeInterfaceIncGen
  )
target_link_libraries(MLIRAffineOps
  PUBLIC
  MLIREDSC
  MLIRIR
  MLIRStandardOps
  )
