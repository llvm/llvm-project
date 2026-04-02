add_mlir_unittest(MLIRPassTests
  AnalysisManagerTest.cpp
  PassManagerTest.cpp
  PassPipelineParserTest.cpp
)
mlir_target_link_libraries(MLIRPassTests
  PRIVATE
  MLIRDebug
  MLIRFuncDialect
  MLIRPass)

# Static pass management unit test infrastructure conflicts.
set_source_files_properties(
  PassManagerTest.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
