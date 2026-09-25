add_lldb_unittest(LLDBPlatformTests
  TestUtils.cpp
  PlatformAppleSimulatorTest.cpp
  PlatformDarwinTest.cpp
  PlatformEmscriptenTest.cpp
  PlatformMacOSXTest.cpp
  PlatformSiginfoTest.cpp
  PlatformTest.cpp
  PlatformWasmTest.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbPluginPlatformEmscripten
    lldbPluginPlatformFreeBSD
    lldbPluginPlatformLinux
    lldbPluginPlatformMacOSX
    lldbPluginPlatformNetBSD
    lldbPluginPlatformWasm
    lldbUtilityHelpers
    LLVMTestingSupport
  )

add_subdirectory(Android)
add_subdirectory(gdb-server)
