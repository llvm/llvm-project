set(LLVM_LINK_COMPONENTS
  Core
  ProfileData
  Support
  )

add_llvm_unittest(ProfileDataTests
  CoverageMappingTest.cpp
  InstrProfTest.cpp
  SampleProfTest.cpp
  )
