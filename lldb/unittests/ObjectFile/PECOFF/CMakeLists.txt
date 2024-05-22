add_lldb_unittest(ObjectFilePECOFFTests
  TestPECallFrameInfo.cpp
  TestSectionSize.cpp

  LINK_LIBS
    lldbUtilityHelpers
    lldbPluginObjectFilePECOFF
    LLVMTestingSupport
  )
