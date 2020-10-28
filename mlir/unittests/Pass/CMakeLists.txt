add_mlir_unittest(MLIRPassTests
  AnalysisManagerTest.cpp
  PassManagerTest.cpp
)
target_link_libraries(MLIRPassTests
  PRIVATE
  MLIRPass)
