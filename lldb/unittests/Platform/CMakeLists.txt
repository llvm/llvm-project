add_lldb_unittest(LLDBPlatformTests
  TestUtils.cpp
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
    lldbUtilityHelpers
    LLVMTestingSupport
  )

add_subdirectory(Android)
add_subdirectory(gdb-server)
