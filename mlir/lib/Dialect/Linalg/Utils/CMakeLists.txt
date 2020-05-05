add_mlir_dialect_library(MLIRLinalgUtils
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Linalg
  )

target_link_libraries(MLIRLinalgUtils
  PUBLIC
  MLIRAffineOps
  MLIREDSC
  MLIRIR
  MLIRLinalgOps
  MLIRLoopOps
  MLIRPass
  MLIRStandardOps
  MLIRTransformUtils
  )
