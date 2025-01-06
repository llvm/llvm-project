add_mlir_unittest(MLIRIRTests
  AdaptorTest.cpp
  AffineExprTest.cpp
  AffineMapTest.cpp
  AttributeTest.cpp
  AttrTypeReplacerTest.cpp
  Diagnostic.cpp
  DialectTest.cpp
  InterfaceTest.cpp
  IRMapping.cpp
  InterfaceAttachmentTest.cpp
  LocationTest.cpp
  OperationSupportTest.cpp
  PatternMatchTest.cpp
  ShapedTypeTest.cpp
  SymbolTableTest.cpp
  TypeTest.cpp
  TypeAttrNamesTest.cpp
  OpPropertiesTest.cpp

  DEPENDS
  MLIRTestInterfaceIncGen
)
target_include_directories(MLIRIRTests PRIVATE "${MLIR_BINARY_DIR}/test/lib/Dialect/Test")
mlir_target_link_libraries(MLIRIRTests PRIVATE MLIRIR)
target_link_libraries(MLIRIRTests PRIVATE MLIRTestDialect)
