add_llvm_unittest(DebuginfodTests
  HTTPClientTests.cpp
  DebuginfodTests.cpp
  )

target_link_libraries(DebuginfodTests PRIVATE
  LLVMDebuginfod
  LLVMTestingSupport
  )
