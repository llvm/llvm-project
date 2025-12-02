set(LLVM_LINK_COMPONENTS
  FuzzerCLI
  Support
)
add_llvm_fuzzer(mlir-bytecode-parser-fuzzer
  mlir-bytecode-parser-fuzzer.cpp
  DUMMY_MAIN DummyParserFuzzer.cpp
)
mlir_target_link_libraries(mlir-bytecode-parser-fuzzer
  PUBLIC
  MLIRIR
  MLIRParser
  MLIRSupport
)

