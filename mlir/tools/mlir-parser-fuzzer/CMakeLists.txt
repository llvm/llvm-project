set(LLVM_LINK_COMPONENTS
  FuzzMutate
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
