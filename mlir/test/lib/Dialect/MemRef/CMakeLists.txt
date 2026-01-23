# Exclude tests from libMLIR.so
add_mlir_library(MLIRMemRefTestPasses
  TestComposeSubView.cpp
  TestEmulateNarrowType.cpp
  TestMultiBuffer.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRTestDialect
  )
mlir_target_link_libraries(MLIRMemRefTestPasses PUBLIC
  MLIRPass
  MLIRMemRefDialect
  MLIRMemRefTransforms
  )

target_include_directories(MLIRMemRefTestPasses
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../Test
  ${CMAKE_CURRENT_BINARY_DIR}/../Test
  )
