add_lldb_unittest(LLDBPlatformTests
  PlatformAppleSimulatorTest.cpp
  PlatformDarwinTest.cpp
  PlatformMacOSXTest.cpp
  PlatformSiginfoTest.cpp
  PlatformTest.cpp

  LINK_LIBS
    lldbPluginPlatformFreeBSD
    lldbPluginPlatformLinux
    lldbPluginPlatformMacOSX
    lldbPluginPlatformNetBSD
  LINK_COMPONENTS
    Support
  )

add_subdirectory(Android)
