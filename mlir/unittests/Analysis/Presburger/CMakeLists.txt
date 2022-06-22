add_mlir_unittest(MLIRPresburgerTests
  IntegerPolyhedronTest.cpp
  IntegerRelationTest.cpp
  LinearTransformTest.cpp
  MatrixTest.cpp
  PresburgerSetTest.cpp
  PresburgerSpaceTest.cpp
  PWMAFunctionTest.cpp
  SimplexTest.cpp
  SlowMPIntTest.cpp
  ../../Dialect/Affine/Analysis/AffineStructuresParser.cpp
)

target_link_libraries(MLIRPresburgerTests
  PRIVATE MLIRPresburger
  MLIRAffineAnalysis
  MLIRParser
  )
