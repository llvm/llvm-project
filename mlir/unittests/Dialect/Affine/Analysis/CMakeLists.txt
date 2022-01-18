add_mlir_unittest(MLIRAffineAnalysisTests
  AffineStructuresParser.cpp
  AffineStructuresParserTest.cpp
)

target_link_libraries(MLIRAffineAnalysisTests
  PRIVATE
  MLIRAffineAnalysis
  MLIRParser
  )
