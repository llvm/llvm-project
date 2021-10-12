add_mlir_dialect_library(MLIRArithmeticTransforms
  Bufferize.cpp
  ExpandOps.cpp

  ADDITIONAL_HEADER_DIRS
  {$MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Arithmetic/Transforms

  DEPENDS
  MLIRArithmeticTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRArithmetic
  MLIRIR
  MLIRMemRef
  MLIRPass
  MLIRTransforms
  )
