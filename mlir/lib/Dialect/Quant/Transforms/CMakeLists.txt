add_mlir_dialect_library(MLIRQuantTransforms
  LowerQuantOps.cpp

  ADDITIONAL_HEADER_DIRS
  {$MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Quant/Transforms

  DEPENDS
  MLIRQuantTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRQuantDialect
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  )
