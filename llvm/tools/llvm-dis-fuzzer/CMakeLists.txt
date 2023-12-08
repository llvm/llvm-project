set(LLVM_LINK_COMPONENTS
  BitReader
)
add_llvm_fuzzer(llvm-dis-fuzzer
  llvm-dis-fuzzer.cpp
  )
