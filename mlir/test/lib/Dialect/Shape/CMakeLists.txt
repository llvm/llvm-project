# Exclude tests from libMLIR.so
add_mlir_library(MLIRShapeTestPasses
  TestShapeFunctions.cpp

  EXCLUDE_FROM_LIBMLIR

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Shape
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRShape
  MLIRSupport
  )
