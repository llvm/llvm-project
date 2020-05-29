add_lldb_unittest(LLDBPlatformTests
  PlatformDarwinTest.cpp

  LINK_LIBS
    lldbPluginPlatformMacOSX
  LINK_COMPONENTS
    Support
  )

add_subdirectory(Android)
