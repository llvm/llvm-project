set(LLVM_LINK_COMPONENTS
  BinaryFormat
  DebugInfoDWARF
  DebugInfoGSYM
  MC
  Support
  ObjectYAML
  )

add_llvm_unittest(DebugInfoGSYMTests
  GsymDataExtractorTest.cpp
  GSYMTest.cpp
  GSYMV2Test.cpp
  )

target_link_libraries(DebugInfoGSYMTests PRIVATE LLVMTestingSupport)
