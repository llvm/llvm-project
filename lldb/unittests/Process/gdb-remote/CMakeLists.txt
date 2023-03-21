add_lldb_unittest(ProcessGdbRemoteTests
  GDBRemoteClientBaseTest.cpp
  GDBRemoteCommunicationClientTest.cpp
  GDBRemoteCommunicationServerLLGSTest.cpp
  GDBRemoteCommunicationServerTest.cpp
  GDBRemoteCommunicationTest.cpp
  GDBRemoteTestUtils.cpp
  PortMapTest.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbPluginPlatformMacOSX
    lldbPluginPlatformLinux
    lldbPluginProcessGDBRemote

    LLVMTestingSupport

  LINK_COMPONENTS
    Support
  )
