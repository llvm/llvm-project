add_mlir_unittest(MLIRPresburgerTests
  FractionTest.cpp
  IntegerPolyhedronTest.cpp
  IntegerRelationTest.cpp
  LinearTransformTest.cpp
  MatrixTest.cpp
  MPIntTest.cpp
  Parser.h
  ParserTest.cpp
  PresburgerSetTest.cpp
  PresburgerRelationTest.cpp
  PresburgerSpaceTest.cpp
  PWMAFunctionTest.cpp
  SimplexTest.cpp
  UtilsTest.cpp
)

target_link_libraries(MLIRPresburgerTests
  PRIVATE MLIRPresburger
  MLIRAffineAnalysis
  MLIRParser
  )
