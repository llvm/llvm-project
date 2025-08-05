add_lldb_unittest(ProtocolServerTests
  ProtocolMCPServerTest.cpp

  LINK_LIBS
    lldbCore
    lldbUtility
    lldbHost
    lldbPluginPlatformMacOSX
    lldbPluginProtocolServerMCP
    LLVMTestingSupport
  )
