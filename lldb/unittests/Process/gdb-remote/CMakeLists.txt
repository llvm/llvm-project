add_lldb_unittest(ProcessGdbRemoteTests
  GDBRemoteClientBaseTest.cpp
  GDBRemoteCommunicationClientTest.cpp
  GDBRemoteCommunicationServerLLGSTest.cpp
  GDBRemoteCommunicationServerTest.cpp
  GDBRemoteCommunicationTest.cpp
  GDBRemoteTestUtils.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbCore
    lldbHost
    lldbPluginPlatformMacOSX
    lldbPluginProcessUtility
    lldbPluginProcessGDBRemote

    LLVMTestingSupport
  )
