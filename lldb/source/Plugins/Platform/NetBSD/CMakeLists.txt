add_lldb_library(lldbPluginPlatformNetBSD PLUGIN
  NetBSDSignals.cpp
  PlatformNetBSD.cpp

   LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbHost
    lldbTarget
  )
