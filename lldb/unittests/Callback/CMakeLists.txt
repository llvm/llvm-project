add_lldb_unittest(LLDBCallbackTests
  TestBreakpointSetCallback.cpp

  LINK_COMPONENTS
    Option
    Support
  LINK_LIBS
    lldbBreakpoint
    lldbCore
    LLVMTestingSupport
    lldbUtilityHelpers
    lldbPluginPlatformMacOSX
  )
