if (CMAKE_SYSTEM_NAME MATCHES "Windows")
  list(APPEND LLDB_WINDOWS_LIBS lldbPluginProcessWindows)
  list(APPEND LLDB_WINDOWS_LIBS lldbPluginProcessWindowsCommon)
endif()

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
      ${LLDB_WINDOWS_LIBS}
  )

