# Exclude tests from libMLIR.so
add_mlir_library(MLIRTestTransforms
  TestConstantFold.cpp
  TestInlining.cpp
  TestLoopFusion.cpp
  TestLoopMapping.cpp
  TestLoopParametricTiling.cpp
  TestLoopUnrolling.cpp

  EXCLUDE_FROM_LIBMLIR

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms

  LINK_LIBS PUBLIC
  MLIRTestDialect
  MLIRTransforms
  )

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Dialect/Test)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../Dialect/Test)
