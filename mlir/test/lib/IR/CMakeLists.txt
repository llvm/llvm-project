# Exclude tests from libMLIR.so
add_mlir_library(MLIRTestIR
  TestAffineWalk.cpp
  TestBytecodeRoundtrip.cpp
  TestBuiltinAttributeInterfaces.cpp
  TestBuiltinDistinctAttributes.cpp
  TestClone.cpp
  TestDiagnostics.cpp
  TestDiagnosticsMetadata.cpp
  TestDominance.cpp
  TestFunc.cpp
  TestInterfaces.cpp
  TestMatchers.cpp
  TestLazyLoading.cpp
  TestOpaqueLoc.cpp
  TestOperationEquals.cpp
  TestPrintDefUse.cpp
  TestPrintInvalid.cpp
  TestPrintNesting.cpp
  TestSideEffects.cpp
  TestSlicing.cpp
  TestSymbolUses.cpp
  TestRegions.cpp
  TestTypes.cpp
  TestUseListOrders.cpp
  TestVisitors.cpp
  TestVisitorsGeneric.cpp

  EXCLUDE_FROM_LIBMLIR

  DEPENDS
  MLIRTestDialect
  )

mlir_target_link_libraries(MLIRTestIR PUBLIC
  MLIRPass
  MLIRBytecodeReader
  MLIRBytecodeWriter
  MLIRFunctionInterfaces
  MLIRTestDialect
  )

target_include_directories(MLIRTestIR
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../Dialect/Test
  ${CMAKE_CURRENT_BINARY_DIR}/../Dialect/Test
  )
