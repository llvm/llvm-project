add_mlir_unittest(MLIRPresburgerTests
  IntegerPolyhedronTest.cpp
  IntegerRelationTest.cpp
  LinearTransformTest.cpp
  MatrixTest.cpp
  MPIntTest.cpp
  Parser.h
  ParserTest.cpp
  PresburgerSetTest.cpp
  PresburgerSpaceTest.cpp
  PWMAFunctionTest.cpp
  SimplexTest.cpp
)

target_link_libraries(MLIRPresburgerTests
  PRIVATE MLIRPresburger
  MLIRAffineAnalysis
  MLIRParser
  )
