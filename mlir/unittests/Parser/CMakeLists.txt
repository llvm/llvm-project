add_mlir_unittest(MLIRParserTests
  ParserTest.cpp
  ResourceTest.cpp

  DEPENDS
  MLIRTestInterfaceIncGen
)
target_include_directories(MLIRParserTests PRIVATE "${MLIR_BINARY_DIR}/test/lib/Dialect/Test")

target_link_libraries(MLIRParserTests PRIVATE
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRIR
  MLIRParser
  MLIRTestDialect
)
