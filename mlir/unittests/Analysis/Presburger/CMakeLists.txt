add_mlir_unittest(MLIRPresburgerTests
  IntegerPolyhedronTest.cpp
  LinearTransformTest.cpp
  MatrixTest.cpp
  PresburgerSetTest.cpp
  PresburgerSpaceTest.cpp
  PWMAFunctionTest.cpp
  SimplexTest.cpp
  ../../Dialect/Affine/Analysis/AffineStructuresParser.cpp
)

target_link_libraries(MLIRPresburgerTests
  PRIVATE MLIRPresburger
  MLIRAffineAnalysis
  MLIRParser
  )
