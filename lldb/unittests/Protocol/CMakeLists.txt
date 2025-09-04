add_lldb_unittest(ProtocolTests
  ProtocolMCPTest.cpp
  ProtocolMCPServerTest.cpp

  LINK_LIBS
    lldbHost
    lldbProtocolMCP
    lldbUtility
    LLVMTestingSupport
  )
