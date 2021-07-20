add_mlir_dialect_library(MLIRLinalgAnalysis
  ConstraintsSet.cpp
  DependenceAnalysis.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Linalg

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRIR
  MLIRLinalg
  MLIRLoopAnalysis
  MLIRMemRef
  MLIRStandard
  )
