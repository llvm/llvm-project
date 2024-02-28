set(LLVM_LINK_COMPONENTS
  Support
  TargetParser
  )

add_llvm_unittest(TargetParserTests
  CSKYTargetParserTest.cpp
  Host.cpp
  TargetParserTest.cpp
  TripleTest.cpp
  RISCVTargetParserTest.cpp
  )

target_link_libraries(TargetParserTests PRIVATE LLVMTestingSupport)
