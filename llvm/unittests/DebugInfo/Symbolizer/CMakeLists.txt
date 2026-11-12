set(LLVM_LINK_COMPONENTS Symbolize)
add_llvm_unittest(DebugInfoSymbolizerTests
  MarkupTest.cpp
  )
target_link_libraries(DebugInfoSymbolizerTests PRIVATE LLVMTestingSupport)
