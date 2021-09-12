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

target_include_directories(MLIRStandardOpsTestPasses
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../Test
  ${CMAKE_CURRENT_BINARY_DIR}/../Test
  )
