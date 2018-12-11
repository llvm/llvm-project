set(LLVM_LINK_COMPONENTS
  TextAPI
)

add_llvm_unittest(TapiTests
  ELFYAMLTest.cpp
)

target_link_libraries(TapiTests PRIVATE LLVMTestingSupport)
