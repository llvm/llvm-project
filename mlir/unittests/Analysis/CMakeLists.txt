add_mlir_unittest(MLIRAnalysisTests
  AffineStructuresTest.cpp
  LinearTransformTest.cpp
  PresburgerSetTest.cpp
)

target_link_libraries(MLIRAnalysisTests
  PRIVATE MLIRLoopAnalysis)

add_subdirectory(Presburger)
