set(LLVM_LINK_COMPONENTS
  DebugInfoGSYM
  MC
  Support
  )

add_llvm_unittest(DebugInfoGSYMTests
  GSYMTest.cpp
  )

target_link_libraries(DebugInfoGSYMTests PRIVATE LLVMTestingSupport)
