add_mlir_dialect_library(MLIRTosaTransforms
  TosaDecomposeTransposeConv.cpp
  TosaDecomposeConv2D.cpp
  TosaDecomposeDepthwise.cpp
  TosaFoldConstantTranspose.cpp
  TosaInferShapes.cpp
  TosaLayerwiseConstantFoldPass.cpp
  TosaMakeBroadcastable.cpp
  TosaOptionalDecompositions.cpp
  TosaValidation.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Tosa/Transforms

  DEPENDS
  MLIRTosaPassIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRPass
  MLIRTosaDialect
  MLIRTransformUtils
  )
