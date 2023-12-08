set(LLVM_LINK_COMPONENTS
  FileCheck
  Support
)

add_llvm_unittest(FileCheckTests
  FileCheckTest.cpp
)

target_link_libraries(FileCheckTests PRIVATE LLVMTestingSupport)
