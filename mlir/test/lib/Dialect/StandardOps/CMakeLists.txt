# Exclude tests from libMLIR.so
add_mlir_library(MLIRStandardOpsTestPasses
  TestDecomposeCallGraphTypes.cpp
  TestComposeSubView.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRPass
  MLIRStandardOpsTransforms
  MLIRTestDialect
  MLIRTransformUtils
  )

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Test)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../Test)
