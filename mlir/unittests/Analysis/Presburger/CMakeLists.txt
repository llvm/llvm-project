add_mlir_unittest(MLIRPresburgerTests
  BarvinokTest.cpp
  FractionTest.cpp
  GeneratingFunctionTest.cpp
  IntegerPolyhedronTest.cpp
  IntegerRelationTest.cpp
  LinearTransformTest.cpp
  MatrixTest.cpp
  ParserTest.cpp
  PresburgerSetTest.cpp
  PresburgerRelationTest.cpp
  PresburgerSpaceTest.cpp
  PWMAFunctionTest.cpp
  QuasiPolynomialTest.cpp
  SimplexTest.cpp
  UtilsTest.cpp
)

target_link_libraries(MLIRPresburgerTests
  PRIVATE MLIRPresburger
  MLIRPresburgerParser
  )
