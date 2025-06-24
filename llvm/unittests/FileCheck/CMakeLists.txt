set(LLVM_LINK_COMPONENTS
  FileCheck
  Support
)

add_llvm_unittest(FileCheckTests
  FileCheckTest.cpp

  DISABLE_LLVM_LINK_LLVM_DYLIB
)

target_link_libraries(FileCheckTests PRIVATE LLVMTestingSupportStatic)
