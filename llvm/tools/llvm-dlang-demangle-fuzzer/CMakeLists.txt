set(LLVM_LINK_COMPONENTS
  Demangle
  FuzzMutate
  Support
)

add_llvm_fuzzer(llvm-dlang-demangle-fuzzer
  llvm-dlang-demangle-fuzzer.cpp
  DUMMY_MAIN DummyDemanglerFuzzer.cpp
  )
