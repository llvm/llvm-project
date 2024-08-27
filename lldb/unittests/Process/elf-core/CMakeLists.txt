add_lldb_unittest(ProcessElfCoreTests
  ThreadElfCoreTest.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbUtilityHelpers
    lldbPluginProcessElfCore
    lldbPluginPlatformLinux

    LLVMTestingSupport

  LINK_COMPONENTS
    Support
  )
