add_mlir_dialect_library(MLIROpenMP
  IR/OpenMPDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/OpenMP
  )

add_dependencies(MLIROpenMP MLIROpenMPOpsIncGen)
target_link_libraries(MLIROpenMP MLIRIR)
