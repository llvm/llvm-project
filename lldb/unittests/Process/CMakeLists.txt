add_subdirectory(gdb-remote)
if (CMAKE_SYSTEM_NAME MATCHES "Linux|Android")
  add_subdirectory(Linux)
  add_subdirectory(POSIX)
endif()
add_subdirectory(Utility)
add_subdirectory(minidump)

add_lldb_unittest(ProcessEventDataTests
  ProcessEventDataTest.cpp

  LINK_LIBS
      lldbCore
      lldbHost
      lldbTarget
      lldbSymbol
      lldbUtility
      lldbUtilityHelpers
      lldbInterpreter
      lldbPluginPlatformMacOSX
  )
