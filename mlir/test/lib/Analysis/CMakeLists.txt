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
  TestTopologicalSort.cpp

  DataFlow/TestDeadCodeAnalysis.cpp
  DataFlow/TestDenseBackwardDataFlowAnalysis.cpp
  DataFlow/TestDenseForwardDataFlowAnalysis.cpp
  DataFlow/TestLivenessAnalysis.cpp
  DataFlow/TestSparseBackwardDataFlowAnalysis.cpp
  DataFlow/TestStridedMetadataRangeAnalysis.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRTestDialect
  )
mlir_target_link_libraries(MLIRTestAnalysis PUBLIC
  MLIRAffineDialect
  MLIRAnalysis
  MLIRFunctionInterfaces
  MLIRMemRefDialect
  MLIRPass
  )

target_include_directories(MLIRTestAnalysis
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../Dialect/Test
  ${CMAKE_CURRENT_BINARY_DIR}/../Dialect/Test
  )
