set(LLVM_LINK_COMPONENTS
  DebugInfoMSF
  )

set(DebugInfoMSFSources
  MappedBlockStreamTest.cpp
  MSFBuilderTest.cpp
  MSFCommonTest.cpp
  )

add_llvm_unittest(DebugInfoMSFTests
  ${DebugInfoMSFSources}
  )

target_link_libraries(DebugInfoMSFTests PRIVATE LLVMTestingSupport)
