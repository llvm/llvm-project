add_lldb_unittest(LLDBServerTests
  TestBase.cpp
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

add_dependencies(LLDBServerTests lldb-server ${ALL_LLDB_TEST_EXECUTABLES})
