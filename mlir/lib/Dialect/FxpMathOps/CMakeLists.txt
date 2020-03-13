add_mlir_dialect_library(MLIRFxpMathOps
  IR/FxpMathOps.cpp
  Transforms/LowerUniformRealMath.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/FxpMathOps

  DEPENDS
  MLIRFxpMathOpsIncGen
  )

target_link_libraries(MLIRFxpMathOps
  PUBLIC
  MLIRQuantOps
  MLIRIR
  MLIRPass
  MLIRSideEffects
  MLIRSupport
  MLIRStandardOps
  )
