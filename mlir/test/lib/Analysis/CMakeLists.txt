# TestAliasAnalysis.cpp doesn't depend on the Test dialect (unlike the rest
# of this directory), so it's split into its own library: downstream users
# that only want to reuse the generic alias-analysis testing boilerplate
# (e.g. ClangIR) can link this without pulling in MLIRTestDialect and
# everything it depends on.
add_mlir_library(MLIRTestAliasAnalysis
  TestAliasAnalysis.cpp

  EXCLUDE_FROM_LIBMLIR
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRFunctionInterfaces
  MLIRIR
  MLIRPass
  )

# Exclude tests from libMLIR.so
add_mlir_library(MLIRTestAnalysis
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
  DataFlow/TestIntegerDivisibilityAnalysis.cpp
  DataFlow/TestLivenessAnalysis.cpp
  DataFlow/TestSparseBackwardDataFlowAnalysis.cpp
  DataFlow/TestStridedMetadataRangeAnalysis.cpp

  EXCLUDE_FROM_LIBMLIR
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRTestAliasAnalysis
  MLIRTestDialect
  )
mlir_target_link_libraries(MLIRTestAnalysis PUBLIC
  MLIRAffineDialect
  MLIRAnalysis
  MLIRArithDialect
  MLIRFunctionInterfaces
  MLIRMemRefDialect
  MLIRPass
  )

target_include_directories(MLIRTestAnalysis
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../Dialect/Test
  ${CMAKE_CURRENT_BINARY_DIR}/../Dialect/Test
  )
