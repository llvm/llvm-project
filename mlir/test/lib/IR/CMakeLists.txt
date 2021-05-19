# Exclude tests from libMLIR.so
add_mlir_library(MLIRTestIR
  TestDominance.cpp
  TestFunc.cpp
  TestInterfaces.cpp
  TestMatchers.cpp
  TestOpaqueLoc.cpp
  TestPrintDefUse.cpp
  TestPrintNesting.cpp
  TestSideEffects.cpp
  TestSlicing.cpp
  TestSymbolUses.cpp
  TestTypes.cpp
  TestVisitors.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTestDialect
  )
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Dialect/Test)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../Dialect/Test)
