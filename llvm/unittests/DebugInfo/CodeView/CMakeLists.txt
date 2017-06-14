set(LLVM_LINK_COMPONENTS
  DebugInfoCodeView
  TestingSupport
  )

set(DebugInfoCodeViewSources
  RandomAccessVisitorTest.cpp
  TypeIndexDiscoveryTest.cpp
  )

add_llvm_unittest(DebugInfoCodeViewTests
  ${DebugInfoCodeViewSources}
  )
