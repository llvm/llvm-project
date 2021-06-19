add_mlir_unittest(MLIRIRTests
  AttributeTest.cpp
  DialectTest.cpp
  InterfaceAttachmentTest.cpp
  MemRefTypeTest.cpp
  OperationSupportTest.cpp
  ShapedTypeTest.cpp
  SubElementInterfaceTest.cpp

  DEPENDS
  MLIRTestInterfaceIncGen
)
target_include_directories(MLIRIRTests PRIVATE "${MLIR_BINARY_DIR}/test/lib/Dialect/Test")
target_link_libraries(MLIRIRTests
  PRIVATE
  MLIRIR
  MLIRTestDialect)
