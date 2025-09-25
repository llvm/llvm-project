add_mlir_unittest(MLIRIRTests
  AdaptorTest.cpp
  AffineExprTest.cpp
  AffineMapTest.cpp
  AttributeTest.cpp
  AttrTypeReplacerTest.cpp
  Diagnostic.cpp
  DialectTest.cpp
  DistinctAttributeAllocatorTest.cpp
  InterfaceTest.cpp
  IRMapping.cpp
  InterfaceAttachmentTest.cpp
  LocationTest.cpp
  MemrefLayoutTest.cpp
  OperationSupportTest.cpp
  PatternMatchTest.cpp
  RemarkTest.cpp  
  ShapedTypeTest.cpp
  SymbolTableTest.cpp
  TypeTest.cpp
  TypeAttrNamesTest.cpp
  OpPropertiesTest.cpp
  ValueTest.cpp
  BlobManagerTest.cpp

  DEPENDS
  MLIRTestInterfaceIncGen
)
target_include_directories(MLIRIRTests PRIVATE "${MLIR_BINARY_DIR}/test/lib/Dialect/Test")
mlir_target_link_libraries(MLIRIRTests PRIVATE MLIRIR)
target_link_libraries(MLIRIRTests PRIVATE MLIRTestDialect)
target_link_libraries(MLIRIRTests PRIVATE MLIRRemarkStreamer)
