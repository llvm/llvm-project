add_mlir_unittest(MLIRParserTests
  ParserTest.cpp
  ResourceTest.cpp

  DEPENDS
  MLIRTestInterfaceIncGen
)

mlir_target_link_libraries(MLIRParserTests PRIVATE
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRIR
  MLIRParser
)
target_link_libraries(MLIRParserTests PRIVATE MLIRTestDialect)
