set(LLVM_LINK_COMPONENTS
  BitReader
  Core
  Coverage
  ProfileData
  Support
  Object
  )

add_llvm_unittest(ProfileDataTests
  BPFunctionNodeTest.cpp
  CoverageMappingTest.cpp
  InstrProfDataTest.cpp
  InstrProfTest.cpp
  ItaniumManglingCanonicalizerTest.cpp
  MemProfTest.cpp
  PGOCtxProfReaderWriterTest.cpp
  SampleProfTest.cpp
  SymbolRemappingReaderTest.cpp
  )

target_link_libraries(ProfileDataTests PRIVATE LLVMTestingSupport)
