set(LLVM_LINK_COMPONENTS
  TestingSupport
  TextAPI
)

add_llvm_unittest(TapiTests
  ELFYAMLTest.cpp
)
