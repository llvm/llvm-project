add_subdirectory(Presburger)

add_mlir_unittest(MLIRAnalysisTests
  SliceWalkTest.cpp
)
mlir_target_link_libraries(MLIRAnalysisTests PRIVATE MLIRAnalysis MLIRIR)
