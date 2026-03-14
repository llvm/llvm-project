add_lldb_unittest(EditlineTests
  EditlineTest.cpp

  LINK_LIBS
    lldbHost
    lldbUtility
    LLVMTestingSupport
  )
