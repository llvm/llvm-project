add_llvm_unittest(LLVMProfgenTests
    ContextCompressionTest.cpp
  )

target_link_libraries(LLVMProfgenTests PRIVATE LLVMTestingSupport)
