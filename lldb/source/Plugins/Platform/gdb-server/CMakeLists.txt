add_lldb_library(lldbPluginPlatformGDB PLUGIN
  GDBRemoteSignals.cpp
  PlatformRemoteGDBServer.cpp

   LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbHost
    lldbTarget
    lldbPluginProcessGDBRemote
  )
