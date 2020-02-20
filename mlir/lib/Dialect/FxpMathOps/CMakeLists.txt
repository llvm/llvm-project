add_mlir_dialect_library(MLIRFxpMathOps
  IR/FxpMathOps.cpp
  Transforms/LowerUniformRealMath.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/FxpMathOps

  DEPENDS
  MLIRFxpMathOpsIncGen

  LINK_LIBS
  MLIRQuantOps
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRStandardOps
  MLIRTransforms
  MLIRTransformUtils
  LLVMSupport
  )
