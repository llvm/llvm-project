add_mlir_dialect_library(MLIRAsyncTransforms
  AsyncParallelFor.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Async

  DEPENDS
  MLIRAsyncPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRAsync
  MLIRSCF
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
)
