set(LLVM_LINK_COMPONENTS
  Support
  FuzzMutate
)

add_llvm_fuzzer(llvm-yaml-parser-fuzzer
  yaml-parser-fuzzer.cpp
  DUMMY_MAIN DummyYAMLParserFuzzer.cpp
  )
