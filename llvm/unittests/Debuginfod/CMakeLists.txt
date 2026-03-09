add_llvm_unittest(DebuginfodTests
  HTTPServerTests.cpp
  DebuginfodTests.cpp
  )

target_link_libraries(DebuginfodTests PRIVATE
  LLVMDebuginfod
  LLVMTestingSupport
  )
