set(LIBS
  MLIRIR
  MLIRStandardOps
  MLIRSupport
  LLVMSupport
  )

add_mlir_dialect_library(MLIRLinalgOps
  LinalgOps.cpp
  LinalgTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Linalg
  DEPENDS
  intrinsics_gen

  LINK_LIBS
  MLIRIR
  MLIRStandardOps
  MLIRSupport
  LLVMSupport
  )

add_dependencies(MLIRLinalgOps

  ${LIBS}
  MLIRLinalgOpsIncGen
  MLIRLinalgStructuredOpsIncGen
  MLIRLinalgStructuredOpsInterfaceIncGen
  )
