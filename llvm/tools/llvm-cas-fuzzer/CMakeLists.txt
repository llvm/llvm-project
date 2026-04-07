set(LLVM_LINK_COMPONENTS
  CAS
  Support
  FuzzerCLI
)

add_llvm_fuzzer(llvm-cas-fuzzer
  cas-fuzzer.cpp
  DUMMY_MAIN DummyCASFuzzer.cpp
)
