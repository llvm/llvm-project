add_mlir_dialect_library(MLIRQuantTransforms
  LowerQuantOps.cpp
  StripFuncQuantTypes.cpp

  ADDITIONAL_HEADER_DIRS
  {$MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Quant/Transforms

  DEPENDS
  MLIRQuantTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRIndexDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgUtils
  MLIRPass
  MLIRQuantDialect
  MLIRShapeDialect
  MLIRTensorDialect
  MLIRTransforms
  MLIRTransformUtils

  )
