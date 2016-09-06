set(LLVM_LINK_COMPONENTS
  Core
  Coverage
  ProfileData
  Support
  )

add_llvm_unittest(ProfileDataTests
  CoverageMappingTest.cpp
  InstrProfTest.cpp
  SampleProfTest.cpp
  )
