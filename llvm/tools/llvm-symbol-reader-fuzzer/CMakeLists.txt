set(LLVM_LINK_COMPONENTS
  ProfileData
)
add_llvm_fuzzer(llvm-symbol-reader-fuzzer
  llvm-symbol-reader-fuzzer.cpp
)
