add_lldb_unittest(LLDBPlatformTests
  PlatformAppleSimulatorTest.cpp
  PlatformDarwinTest.cpp
  PlatformMacOSXTest.cpp
  PlatformSiginfoTest.cpp
  PlatformTest.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbPluginPlatformFreeBSD
    lldbPluginPlatformLinux
    lldbPluginPlatformMacOSX
    lldbPluginPlatformNetBSD
  )

add_subdirectory(Android)
add_subdirectory(gdb-server)
