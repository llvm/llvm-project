if (CMAKE_SYSTEM_NAME MATCHES "Windows")
  list(APPEND LLDB_WINDOWS_LIBS lldbPluginPlatformWindows)
  list(APPEND LLDB_WINDOWS_LIBS lldbPluginProcessWindowsCommon)
  list(APPEND LLDB_WINDOWS_LIBS lldbPluginDynamicLoaderWindowsDYLD)
  list(APPEND LLDB_WINDOWS_LIBS lldbPluginObjectFilePECOFF)
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

