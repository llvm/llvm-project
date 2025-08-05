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
    AArch32ErrorTests.cpp
    AArch64Tests.cpp
    EHFrameSupportTests.cpp
    JITLinkTestUtils.cpp
    LinkGraphTests.cpp
    MachOLinkGraphTests.cpp
    MemoryManagerErrorTests.cpp
    StubsTests.cpp
    X86_64Tests.cpp
  )

target_link_libraries(JITLinkTests PRIVATE LLVMTestingSupport)
