set(LLVM_LINK_COMPONENTS
  FuzzerCLI
  Support
)
add_llvm_fuzzer(mlir-parser-fuzzer
  mlir-parser-fuzzer.cpp
  DUMMY_MAIN DummyParserFuzzer.cpp
)
target_link_libraries(mlir-parser-fuzzer
  PUBLIC
  MLIRIR
  MLIRParser
  MLIRSupport
)
