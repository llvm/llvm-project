set(LLVM_LINK_COMPONENTS
  DebugInfoMSF
  )

set(LLVM_OPTIONAL_SOURCES
  MSFZTest.cpp
  )

if(LLVM_ENABLE_ZSTD)
  set(msfz_test MSFZTest.cpp)
endif()

add_llvm_unittest(DebugInfoMSFTests
  MappedBlockStreamTest.cpp
  MSFBuilderTest.cpp
  MSFCommonTest.cpp
  ${msfz_test}
  )

target_link_libraries(DebugInfoMSFTests PRIVATE LLVMTestingSupport)
