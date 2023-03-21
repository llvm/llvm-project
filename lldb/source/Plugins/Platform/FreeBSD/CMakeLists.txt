add_lldb_library(lldbPluginPlatformFreeBSD PLUGIN
  FreeBSDSignals.cpp
  PlatformFreeBSD.cpp

   LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbHost
    lldbTarget

  LINK_COMPONENTS
    TargetParser
  )
