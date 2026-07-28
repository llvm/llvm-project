add_lldb_unittest(ObjectFilePECOFFTests
  TestCompressedSection.cpp
  TestPECallFrameInfo.cpp
  TestSectionSize.cpp

  LINK_LIBS
    lldbUtilityHelpers
    lldbPluginObjectFilePECOFF
    LLVMTestingSupport
  )
