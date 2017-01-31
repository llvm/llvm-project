include_directories(.)
include_directories(../POSIX)
include_directories(../Utility)

add_lldb_library(lldbPluginProcessFreeBSD PLUGIN
  ProcessFreeBSD.cpp
  FreeBSDThread.cpp
  ProcessMonitor.cpp

  POSIXStopInfo.cpp
  RegisterContextPOSIXProcessMonitor_arm.cpp
  RegisterContextPOSIXProcessMonitor_arm64.cpp
  RegisterContextPOSIXProcessMonitor_powerpc.cpp
  RegisterContextPOSIXProcessMonitor_x86.cpp
  RegisterContextPOSIXProcessMonitor_mips64.cpp

  LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbHost
    lldbSymbol
    lldbTarget
    lldbUtility
    lldbPluginProcessUtility
    lldbPluginProcessPOSIX
  LINK_COMPONENTS
    Support
  )
