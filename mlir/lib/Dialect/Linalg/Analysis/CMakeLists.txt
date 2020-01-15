set(LIBS

  MLIRLinalgOps
  MLIRStandardOps
  )


add_llvm_library(MLIRLinalgAnalysis
  DependenceAnalysis.cpp
  
  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Linalg
  DEPENDS
  intrinsics_gen
  )

add_dependencies(MLIRLinalgAnalysis ${LIBS})
target_link_libraries(MLIRLinalgAnalysis ${LIBS})
