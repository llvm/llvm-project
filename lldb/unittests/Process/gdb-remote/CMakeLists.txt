add_lldb_unittest(ProcessGdbRemoteTests
  GDBRemoteClientBaseTest.cpp
  GDBRemoteCommunicationClientTest.cpp
  GDBRemoteCommunicationTest.cpp
  GDBRemoteTestUtils.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbPluginPlatformMacOSX
    lldbPluginProcessUtility
    lldbPluginProcessGDBRemote

    LLVMTestingSupport

  LINK_COMPONENTS
    Support
  )
