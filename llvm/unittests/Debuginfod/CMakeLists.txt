set(LLVM_LINK_COMPONENTS
  SupportHTTP
  )

add_llvm_unittest(DebuginfodTests
  DebuginfodTests.cpp
  )

target_link_libraries(DebuginfodTests PRIVATE
  LLVMDebuginfod
  LLVMTestingSupport
  )
