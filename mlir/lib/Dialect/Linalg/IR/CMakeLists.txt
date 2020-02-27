set(LIBS

  MLIRIR
  MLIRStandardOps
  )

add_llvm_library(MLIRLinalgOps
  LinalgOps.cpp
  LinalgTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Linalg
  DEPENDS
  intrinsics_gen
  )

add_dependencies(MLIRLinalgOps

  ${LIBS}
  MLIRLinalgOpsIncGen
  MLIRLinalgStructuredOpsIncGen
  MLIRLinalgStructuredOpsInterfaceIncGen
  )
target_link_libraries(MLIRLinalgOps ${LIBS})
