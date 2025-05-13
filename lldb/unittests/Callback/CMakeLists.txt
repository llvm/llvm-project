add_lldb_unittest(LLDBCallbackTests
  TestBreakpointSetCallback.cpp

  LINK_LIBS
    lldbBreakpoint
    lldbCore
    LLVMTestingSupport
    lldbUtilityHelpers
    lldbPluginPlatformMacOSX
  LINK_COMPONENTS
    Support
  )
