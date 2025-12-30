add_lldb_unittest(LLDBCallbackTests
  TestBreakpointSetCallback.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbBreakpoint
    lldbCore
    LLVMTestingSupport
    lldbUtilityHelpers
    lldbPluginPlatformMacOSX
  )
