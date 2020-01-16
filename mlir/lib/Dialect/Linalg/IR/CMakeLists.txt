set(LIBS

  MLIRIR
  )

add_llvm_library(MLIRLinalgOps
  LinalgOps.cpp
  LinalgTypes.cpp
  LinalgRegistration.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Linalg
  DEPENDS
  intrinsics_gen
  )

add_dependencies(MLIRLinalgOps

  ${LIBS}
  MLIRLinalgOpsIncGen
  MLIRLinalgStructuredOpsIncGen
  )
target_link_libraries(MLIRLinalgOps ${LIBS})
