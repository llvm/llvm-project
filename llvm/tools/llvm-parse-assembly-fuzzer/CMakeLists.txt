set(LLVM_LINK_COMPONENTS
  AsmParser
)
add_llvm_fuzzer(llvm-parse-assembly-fuzzer
  llvm-parse-assembly-fuzzer.cpp
)
