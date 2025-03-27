add_lldb_unittest(ProcessGdbRemoteTests
  GDBRemoteClientBaseTest.cpp
  GDBRemoteCommunicationClientTest.cpp
  GDBRemoteCommunicationServerLLGSTest.cpp
  GDBRemoteCommunicationServerTest.cpp
  GDBRemoteCommunicationTest.cpp
  GDBRemoteTestUtils.cpp

  LINK_LIBS
    LLVMTestingSupport
    lldbCore
    lldbHost
    lldbInterpreter
    lldbPluginProcessUtility
    lldbSymbol
    lldbTarget
    lldbValueObject

  LINK_COMPONENTS
    Support
  )
