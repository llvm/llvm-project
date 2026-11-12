set(LLVM_LINK_COMPONENTS
  Core
  ProfileData
  Support
  )

add_llvm_unittest(LLVMProfdataTests
    OutputSizeLimitTest.cpp
  )

target_link_libraries(LLVMProfdataTests PRIVATE LLVMTestingSupport)
