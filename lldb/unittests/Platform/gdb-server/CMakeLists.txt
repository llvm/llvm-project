add_lldb_unittest(PlatformGdbRemoteTests
  PlatformRemoteGDBServerTest.cpp

  LINK_LIBS
    lldbPluginPlatformGDB
    LLVMTestingSupport
  )
