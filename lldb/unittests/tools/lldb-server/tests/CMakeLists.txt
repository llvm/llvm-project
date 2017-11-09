add_lldb_unittest(LLDBServerTests
  TestClient.cpp
  MessageObjects.cpp
  ThreadIdsInJstopinfoTest.cpp

  LINK_LIBS
    lldbHost
    lldbCore
    lldbInterpreter
    lldbTarget
    lldbPluginPlatformLinux
    lldbPluginProcessGDBRemote

    LLVMTestingSupport
  LINK_COMPONENTS
    Support
  )
