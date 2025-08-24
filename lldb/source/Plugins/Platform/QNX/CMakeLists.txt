add_lldb_library(lldbPluginPlatformQNX PLUGIN
  PlatformQNX.cpp

   LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbHost
    lldbInterpreter
    lldbTarget
    lldbPluginPlatformPOSIX
  )
