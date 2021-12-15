add_mlir_unittest(MLIRPresburgerTests
  IntegerPolyhedronTest.cpp
  MatrixTest.cpp
  SimplexTest.cpp
  ../AffineStructuresParser.cpp
)

target_link_libraries(MLIRPresburgerTests
  PRIVATE MLIRPresburger
  MLIRLoopAnalysis
  MLIRParser
  )
