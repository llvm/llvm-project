set(LLVM_LINK_COMPONENTS
  Demangle
  FuzzMutate
)

add_llvm_fuzzer(llvm-demangle-fuzzer
  llvm-demangle-fuzzer.cpp
  DUMMY_MAIN DummyDemanglerFuzzer.cpp)
