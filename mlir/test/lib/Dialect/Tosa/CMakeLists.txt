add_mlir_dialect_library(MLIRTosaTestPasses
  TosaTestPasses.cpp

  EXCLUDE_FROM_LIBMLIR

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Tosa/Transforms

  DEPENDS
  MLIRTosaPassIncGen

  LINK_LIBS PUBLIC
  MLIRFunc
  MLIRPass
  MLIRTosa
  MLIRTransformUtils
  )
