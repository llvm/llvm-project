set(LLVM_LINK_COMPONENTS
  Support
  TargetParser
  )

add_llvm_unittest(TargetParserTests
  AtomicScopeTest.cpp
  CSKYTargetParserTest.cpp
  Host.cpp
  NVPTXTargetParserTest.cpp
  RISCVISAInfoTest.cpp
  RISCVTargetParserTest.cpp
  TargetParserTest.cpp
  TripleTest.cpp
  )

target_link_libraries(TargetParserTests PRIVATE LLVMTestingSupport)
