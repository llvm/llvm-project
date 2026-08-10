set(LLVM_LINK_COMPONENTS
  Object
  ObjectYAML
)
add_llvm_fuzzer(llvm-object-yaml-fuzzer
  llvm-object-yaml-fuzzer.cpp
)
