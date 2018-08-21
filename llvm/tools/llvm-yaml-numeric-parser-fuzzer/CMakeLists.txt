set(LLVM_LINK_COMPONENTS
  Support
  FuzzMutate
)

add_llvm_fuzzer(llvm-yaml-numeric-parser-fuzzer
  yaml-numeric-parser-fuzzer.cpp
  DUMMY_MAIN DummyYAMLNumericParserFuzzer.cpp
  )
