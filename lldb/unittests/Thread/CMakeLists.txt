add_lldb_unittest(ThreadTests
  ThreadTest.cpp

  LINK_LIBS
      lldbCore
      lldbHost
      lldbTarget
      lldbSymbol
      lldbUtility
      lldbUtilityHelpers
      lldbInterpreter
      lldbBreakpoint
      lldbPluginPlatformLinux
  )

