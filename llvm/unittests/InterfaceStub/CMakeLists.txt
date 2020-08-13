set(LLVM_LINK_COMPONENTS
  InterfaceStub
)

add_llvm_unittest(InterfaceStubTests
  ELFYAMLTest.cpp
)

target_link_libraries(InterfaceStubTests PRIVATE LLVMTestingSupport)
