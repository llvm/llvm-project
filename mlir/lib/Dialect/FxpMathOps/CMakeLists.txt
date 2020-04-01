add_mlir_dialect_library(MLIRFxpMathOps
  IR/FxpMathOps.cpp
  Transforms/LowerUniformRealMath.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/FxpMathOps

  DEPENDS
  MLIRFxpMathOpsIncGen
  MLIRFxpMathPassIncGen
  )

target_link_libraries(MLIRFxpMathOps
  PUBLIC
  MLIRQuant
  MLIRIR
  MLIRPass
  MLIRSideEffects
  MLIRSupport
  MLIRStandardOps
  )
