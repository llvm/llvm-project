add_lldb_unittest(ProtocolTests
  ProtocolMCPTest.cpp
  ProtocolMCPServerTest.cpp

  LINK_LIBS
    lldbCore
    lldbUtility
    lldbHost
    lldbPluginPlatformMacOSX
    lldbPluginProtocolServerMCP
    LLVMTestingSupport
  )
