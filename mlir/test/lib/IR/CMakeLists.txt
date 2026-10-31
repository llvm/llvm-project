# Exclude tests from libMLIR.so
add_mlir_library(MLIRTestIR
  TestAffineWalk.cpp
  TestBytecodeRoundtrip.cpp
  TestBuiltinAttributeInterfaces.cpp
  TestBuiltinDistinctAttributes.cpp
  TestClone.cpp
  TestDiagnostics.cpp
  TestDiagnosticsMetadata.cpp
  TestDiagnosticsNext.cpp
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

  LINK_LIBS PUBLIC
  MLIRTestDialect
  )

mlir_target_link_libraries(MLIRTestIR PUBLIC
  MLIRAnalysisBase
  MLIRBytecodeReader
  MLIRBytecodeWriter
  MLIRFunctionInterfaces
  MLIRPass
  )
