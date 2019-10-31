set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  JITLink
  Object
  RuntimeDyld
  Support
  )

add_llvm_unittest(JITLinkTests
    LinkGraphTests.cpp
  )

target_link_libraries(JITLinkTests PRIVATE LLVMTestingSupport)
