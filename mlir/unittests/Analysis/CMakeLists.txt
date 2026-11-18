add_mlir_unittest(MLIRAnalysisTests
  LivenessTest.cpp
)

mlir_target_link_libraries(MLIRAnalysisTests
  PRIVATE MLIRAnalysis
  MLIRParser
  )

add_subdirectory(Presburger)
