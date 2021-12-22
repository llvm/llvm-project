add_mlir_unittest(MLIRAnalysisTests
  AffineStructuresParser.cpp
  AffineStructuresParserTest.cpp
  AffineStructuresTest.cpp
  LinearTransformTest.cpp
  PresburgerSetTest.cpp
)

target_link_libraries(MLIRAnalysisTests
  PRIVATE
  MLIRLoopAnalysis
  MLIRParser
  )

add_subdirectory(Presburger)
