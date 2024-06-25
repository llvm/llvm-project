add_lldb_unittest(LLDBBreakpointTests
  BreakpointIDTest.cpp
  WatchpointAlgorithmsTests.cpp

  LINK_LIBS
    lldbBreakpoint
    lldbCore
  LINK_COMPONENTS
    Support
  )
