add_lldb_unittest(ObjectFilePECOFFTests
  TestPECallFrameInfo.cpp

  LINK_LIBS
    lldbUtilityHelpers
    lldbPluginObjectFilePECOFF
    LLVMTestingSupport
  )
