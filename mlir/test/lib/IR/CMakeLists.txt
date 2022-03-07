# Exclude tests from libMLIR.so
add_mlir_library(MLIRTestIR
  TestBuiltinAttributeInterfaces.cpp
  TestDiagnostics.cpp
  TestDominance.cpp
  TestFunc.cpp
  TestInterfaces.cpp
  TestMatchers.cpp
  TestOpaqueLoc.cpp
  TestOperationEquals.cpp
  TestPrintDefUse.cpp
  TestPrintInvalid.cpp
  TestPrintNesting.cpp
  TestSideEffects.cpp
  TestSlicing.cpp
  TestSymbolUses.cpp
  TestTypes.cpp
  TestVisitors.cpp
  TestVisitorsGeneric.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTestDialect
  )

target_include_directories(MLIRTestIR
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../Dialect/Test
  ${CMAKE_CURRENT_BINARY_DIR}/../Dialect/Test
  )
