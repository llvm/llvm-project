add_lldb_unittest(ObjectFilePECOFFTests
  TestMemoryInstance.cpp
  TestPECallFrameInfo.cpp
  TestSectionList.cpp
  TestSectionSize.cpp

  LINK_LIBS
    lldbUtilityHelpers
    lldbPluginObjectFilePECOFF
    LLVMTestingSupport
  )
