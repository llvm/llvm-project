# Exclude tests from libMLIR.so
add_mlir_library(MLIRTestAnalysis
  TestAliasAnalysis.cpp
  TestCallGraph.cpp
  TestDataFlowFramework.cpp
  TestLiveness.cpp
  TestCFGLoopInfo.cpp
  TestMatchReduction.cpp
  TestMemRefBoundCheck.cpp
  TestMemRefDependenceCheck.cpp
  TestMemRefStrideCalculation.cpp
  TestSlice.cpp

  DataFlow/TestDeadCodeAnalysis.cpp
  DataFlow/TestDenseBackwardDataFlowAnalysis.cpp
  DataFlow/TestDenseForwardDataFlowAnalysis.cpp
  DataFlow/TestLivenessAnalysis.cpp
  DataFlow/TestSparseBackwardDataFlowAnalysis.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAnalysis
  MLIRFunctionInterfaces
  MLIRMemRefDialect
  MLIRPass
  MLIRTestDialect
  )

target_include_directories(MLIRTestAnalysis
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../Dialect/Test
  ${CMAKE_CURRENT_BINARY_DIR}/../Dialect/Test
  )
