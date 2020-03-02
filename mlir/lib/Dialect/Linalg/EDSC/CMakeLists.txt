set(LIBS

  MLIREDSC
  MLIRIR
  MLIRAffineOps
  MLIRLinalgOps
  MLIRLoopOps
  MLIRStandardOps
  )

add_mlir_dialect_library(MLIRLinalgEDSC
  Builders.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Linalg
  DEPENDS
  intrinsics_gen
  )

add_dependencies(MLIRLinalgEDSC ${LIBS})
target_link_libraries(MLIRLinalgEDSC ${LIBS})
