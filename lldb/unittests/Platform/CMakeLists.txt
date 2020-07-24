add_lldb_unittest(LLDBPlatformTests
  PlatformAppleSimulatorTest.cpp
  PlatformDarwinTest.cpp

  LINK_LIBS
    lldbPluginPlatformMacOSX
  LINK_COMPONENTS
    Support
  )
