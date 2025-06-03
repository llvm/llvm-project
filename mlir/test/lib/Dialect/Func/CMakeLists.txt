# Exclude tests from libMLIR.so
add_mlir_library(MLIRFuncTestPasses
  TestDecomposeCallGraphTypes.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRTestDialect
  )
mlir_target_link_libraries(MLIRFuncTestPasses PUBLIC
  MLIRAffineDialect
  MLIRPass
  MLIRFuncTransforms
  MLIRTransformUtils
  )

target_include_directories(MLIRFuncTestPasses
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../Test
  ${CMAKE_CURRENT_BINARY_DIR}/../Test
  )
