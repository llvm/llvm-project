set(LLVM_LINK_COMPONENTS
  DebugInfoMSF
  )

add_llvm_unittest(DebugInfoMSFTests
  MappedBlockStreamTest.cpp
  MSFBuilderTest.cpp
  MSFCommonTest.cpp
  )

target_link_libraries(DebugInfoMSFTests PRIVATE LLVMTestingSupport)

set_property(TARGET DebugInfoMSFTests PROPERTY FOLDER "Tests/UnitTests/DebugInfoTests")
