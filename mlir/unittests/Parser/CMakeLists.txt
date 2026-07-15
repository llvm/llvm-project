add_mlir_unittest(MLIRParserTests
  ParserTest.cpp
  ResourceTest.cpp

  DEPENDS
  MLIRTestInterfaceIncGen
)
target_include_directories(MLIRParserTests PRIVATE "${MLIR_BINARY_DIR}/test/lib/Dialect/Test")

mlir_target_link_libraries(MLIRParserTests PRIVATE
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRIR
  MLIRParser
)
target_link_libraries(MLIRParserTests PRIVATE MLIRTestDialect)
