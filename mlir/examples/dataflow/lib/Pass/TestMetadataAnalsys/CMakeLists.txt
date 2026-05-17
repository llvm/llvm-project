add_mlir_example_library(MLIRTestMetadataAnalysisPass
  TestMetadataAnalsys.cpp

  LINK_LIBS PUBLIC
  MLIRMetadataAnalysis
)
