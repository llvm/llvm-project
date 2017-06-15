set(LLVM_LINK_COMPONENTS
  DebugInfoCodeView
  )

set(DebugInfoCodeViewSources
  RandomAccessVisitorTest.cpp
  TypeIndexDiscoveryTest.cpp
  )

add_llvm_unittest(DebugInfoCodeViewTests
  ${DebugInfoCodeViewSources}
  )

target_link_libraries(DebugInfoCodeViewTests LLVMTestingSupport)