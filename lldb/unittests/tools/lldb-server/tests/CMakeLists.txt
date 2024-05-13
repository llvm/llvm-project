add_lldb_unittest(LLDBServerTests
  LLGSTest.cpp
  MessageObjects.cpp
  TestBase.cpp
  TestClient.cpp

  LINK_LIBS
    lldbHost
    lldbCore
    lldbHostHelpers
    lldbInterpreter
    lldbTarget
    lldbPluginPlatformLinux
    lldbPluginProcessGDBRemote

    LLVMTestingSupport
  LINK_COMPONENTS
    Support
  )

add_dependencies(LLDBServerTests lldb-server ${ALL_LLDB_TEST_EXECUTABLES})
