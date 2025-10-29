add_subdirectory(gdb-remote)
if (CMAKE_SYSTEM_NAME MATCHES "Linux|Android")
  add_subdirectory(elf-core)
  add_subdirectory(Linux)
  add_subdirectory(POSIX)
endif()
add_subdirectory(Utility)
add_subdirectory(minidump)

add_lldb_unittest(ProcessTests
  ProcessEventDataTest.cpp
  ProcessTraceTest.cpp

  LINK_LIBS
      lldbCore
      lldbHost
      lldbTarget
      lldbSymbol
      lldbUtility
      lldbUtilityHelpers
      lldbInterpreter
      lldbPluginPlatformLinux
      lldbPluginPlatformMacOSX
  )
