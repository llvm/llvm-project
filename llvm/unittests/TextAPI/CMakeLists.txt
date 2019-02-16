set(LLVM_LINK_COMPONENTS
  TextAPI
)

add_llvm_unittest(TextAPITests
  ELFYAMLTest.cpp
)

target_link_libraries(TextAPITests PRIVATE LLVMTestingSupport)
