add_definitions("-D_ALL_SOURCE")

add_lldb_library(lldbPluginPlatformAIX PLUGIN
  PlatformAIX.cpp

   LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbHost
    lldbInterpreter
    lldbTarget
    lldbPluginPlatformPOSIX
  )
