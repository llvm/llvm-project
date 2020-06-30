# Exclude tests from libMLIR.so
add_mlir_library(MLIRTestIR
  TestFunc.cpp
  TestInterfaces.cpp
  TestMatchers.cpp
  TestSideEffects.cpp
  TestSymbolUses.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTestDialect
  )
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Dialect/Test)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../Dialect/Test)
