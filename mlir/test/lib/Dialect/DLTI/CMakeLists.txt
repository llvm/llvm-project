# Exclude tests from libMLIR.so
add_mlir_library(MLIRDLTITestPasses
  TestDataLayoutQuery.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRDLTI
  MLIRPass
  MLIRTestDialect
  )

target_include_directories(MLIRDLTITestPasses
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../Test
  ${CMAKE_CURRENT_BINARY_DIR}/../Test
  )
