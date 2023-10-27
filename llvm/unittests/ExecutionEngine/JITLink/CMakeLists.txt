set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  JITLink
  Object
  OrcShared
  Support
  TargetParser
  )

add_llvm_unittest(JITLinkTests
    AArch32Tests.cpp
    EHFrameSupportTests.cpp
    JITLinkMocks.cpp
    LinkGraphTests.cpp
    MemoryManagerErrorTests.cpp
    StubsTests.cpp
  )

target_link_libraries(JITLinkTests PRIVATE LLVMTestingSupport)

set_property(TARGET JITLinkTests PROPERTY FOLDER "Tests/UnitTests/ExecutionTests")
