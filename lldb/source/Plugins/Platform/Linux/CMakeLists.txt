add_lldb_library(lldbPluginPlatformLinux PLUGIN
  LinuxSignals.cpp
  PlatformLinux.cpp

   LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbHost
    lldbInterpreter
    lldbTarget
    lldbPluginPlatformPOSIX
  )
