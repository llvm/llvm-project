add_mlir_unittest(MLIRIRTests
  AdaptorTest.cpp
  AttributeTest.cpp
  BlockAndValueMapping.cpp
  DialectTest.cpp
  InterfaceTest.cpp
  InterfaceAttachmentTest.cpp
  OperationSupportTest.cpp
  PatternMatchTest.cpp
  ShapedTypeTest.cpp
  SubElementInterfaceTest.cpp
  TypeTest.cpp

  DEPENDS
  MLIRTestInterfaceIncGen
)
target_include_directories(MLIRIRTests PRIVATE "${MLIR_BINARY_DIR}/test/lib/Dialect/Test")
target_link_libraries(MLIRIRTests
  PRIVATE
  MLIRIR
  MLIRTestDialect)
