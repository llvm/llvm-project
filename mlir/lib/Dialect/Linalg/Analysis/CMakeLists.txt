add_mlir_dialect_library(MLIRLinalgAnalysis
  DependenceAnalysis.cpp
  
  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Linalg

  DEPENDS
  intrinsics_gen

  LINK_LIBS
  MLIRIR
  MLIRLinalgOps
  MLIRStandardOps
  LLVMSupport
  )
