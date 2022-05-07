set(LLVM_LINK_COMPONENTS
  Demangle
  FuzzMutate
  Support
)

add_llvm_fuzzer(llvm-microsoft-demangle-fuzzer
  llvm-microsoft-demangle-fuzzer.cpp
  DUMMY_MAIN DummyDemanglerFuzzer.cpp
  )
