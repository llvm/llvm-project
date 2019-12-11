set(LLVM_LINK_COMPONENTS
  Analysis
  Core
  FrontendOpenMP
  Support
  Passes
  )

add_llvm_unittest(LLVMFrontendTests
  OpenMPIRBuilderTest.cpp
  )

target_link_libraries(LLVMFrontendTests PRIVATE LLVMTestingSupport)
