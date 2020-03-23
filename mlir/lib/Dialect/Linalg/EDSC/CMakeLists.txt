add_mlir_dialect_library(MLIRLinalgEDSC
  Builders.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Linalg

  DEPENDS
  intrinsics_gen
  )

target_link_libraries(MLIRLinalgEDSC
  PUBLIC
  MLIREDSC
  MLIRIR
  MLIRAffine
  MLIRLinalgOps
  MLIRLoopOps
  MLIRStandardOps
  LLVMSupport
  )
