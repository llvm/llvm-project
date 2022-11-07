set(LLVM_LINK_COMPONENTS
  FuzzerCLI
  Support
)
add_llvm_fuzzer(mlir-text-parser-fuzzer
  mlir-text-parser-fuzzer.cpp
  DUMMY_MAIN DummyParserFuzzer.cpp
)
target_link_libraries(mlir-text-parser-fuzzer
  PUBLIC
  MLIRIR
  MLIRParser
  MLIRSupport
)

