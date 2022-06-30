add_mlir_unittest(MLIRParserTests
  ResourceTest.cpp

  DEPENDS
  MLIRTestInterfaceIncGen
)
target_include_directories(MLIRParserTests PRIVATE "${MLIR_BINARY_DIR}/test/lib/Dialect/Test")

target_link_libraries(MLIRParserTests PRIVATE
  MLIRIR
  MLIRParser
  MLIRTestDialect
)
