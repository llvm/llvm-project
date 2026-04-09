add_llvm_unittest(DebuginfodTests
  DebuginfodTests.cpp
  )

target_link_libraries(DebuginfodTests PRIVATE
  LLVMDebuginfod
  LLVMTestingSupport
  )
