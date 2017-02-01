add_lldb_library(lldbPluginPlatformFreeBSD PLUGIN
  PlatformFreeBSD.cpp

   LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbHost
    lldbTarget
  )
