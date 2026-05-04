add_lldb_library(lldbPluginPlatformFreeBSD PLUGIN
  PlatformFreeBSD.cpp

  LINK_COMPONENTS
    TargetParser
   LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbHost
    lldbTarget
    lldbPluginPlatformPOSIX
  )
