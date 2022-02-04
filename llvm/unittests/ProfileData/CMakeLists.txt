set(LLVM_LINK_COMPONENTS
  Core
  Coverage
  ProfileData
  Support
  )

add_llvm_unittest(ProfileDataTests
  CoverageMappingTest.cpp
  InstrProfDataTest.cpp
  InstrProfTest.cpp
  SampleProfTest.cpp
  MemProfTest.cpp
  )

target_link_libraries(ProfileDataTests PRIVATE LLVMTestingSupport)
