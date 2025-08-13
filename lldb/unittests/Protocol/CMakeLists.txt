add_lldb_unittest(ProtocolTests
  ProtocolMCPTest.cpp

  LINK_LIBS
    lldbHost
    lldbProtocolMCP
    lldbUtility
    LLVMTestingSupport
  )
