set(LLVM_LINK_COMPONENTS
  Support
  TargetParser
  )

add_llvm_unittest(TargetParserTests
  CSKYTargetParserTest.cpp
  Host.cpp
  RISCVISAInfoTest.cpp
  RISCVTargetParserTest.cpp
  TargetParserTest.cpp
  TripleTest.cpp
  )

target_link_libraries(TargetParserTests PRIVATE LLVMTestingSupport)
