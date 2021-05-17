set(LLVM_LINK_COMPONENTS
  Demangle
  FuzzMutate
  Support
)

add_llvm_fuzzer(llvm-rust-demangle-fuzzer
  llvm-rust-demangle-fuzzer.cpp
  DUMMY_MAIN DummyDemanglerFuzzer.cpp
  )
