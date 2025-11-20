add_mlir_dialect_library(MLIRAffineAnalysis
  AffineAnalysis.cpp
  AffineStructures.cpp
  LoopAnalysis.cpp
  NestedMatcher.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Affine

  DEPENDS
  MLIRFuncOpsIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAnalysis
  MLIRCallInterfaces
  MLIRControlFlowInterfaces
  MLIRDialectUtils
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
  MLIRPresburger
  MLIRSCFDialect
  )
