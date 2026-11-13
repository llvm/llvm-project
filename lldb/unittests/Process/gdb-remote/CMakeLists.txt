add_lldb_unittest(ProcessGdbRemoteTests
  GDBRemoteClientBaseTest.cpp
  GDBRemoteCommunicationClientTest.cpp
  GDBRemoteCommunicationServerLLGSTest.cpp
  GDBRemoteCommunicationServerTest.cpp
  GDBRemoteCommunicationTest.cpp
  GDBRemoteTestUtils.cpp
  ProcessGDBRemoteTest.cpp
  RegisterPacketLayoutTest.cpp

  LINK_COMPONENTS
    Option
    Support
  LINK_LIBS
    lldbCore
    lldbHost
    lldbPluginPlatformMacOSX
    lldbPluginProcessGDBRemote
    lldbPluginProcessUtility
    lldbRegister

    LLVMTestingSupport
  )
