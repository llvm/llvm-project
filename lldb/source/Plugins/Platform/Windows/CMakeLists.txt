add_lldb_library(lldbPluginPlatformWindows PLUGIN
  PlatformWindows.cpp

   LINK_COMPONENTS
    Support
   LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbHost
    lldbTarget
    lldbPluginPlatformGDB
  )
