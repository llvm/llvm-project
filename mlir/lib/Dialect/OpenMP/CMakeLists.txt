add_mlir_dialect_library(MLIROpenMP
  IR/OpenMPDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/OpenMP

  DEPENDS
  MLIROpenMPOpsIncGen
  MLIROpenMPOpsInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMIR
  )
