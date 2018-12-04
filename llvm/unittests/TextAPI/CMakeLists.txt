set(LLVM_LINK_COMPONENTS
  TextAPI
)

add_llvm_unittest(TapiTests
  ELFYAMLTest.cpp
)
