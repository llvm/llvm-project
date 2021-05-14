# Exclude tests from libMLIR.so
add_mlir_library(MLIRDLTITestPasses
  TestDataLayoutQuery.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRDLTI
  MLIRPass
  MLIRTestDialect
  )

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Test)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../Test)
