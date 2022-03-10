add_lldb_library(lldbPluginPlatformWindows PLUGIN
  PlatformWindows.cpp

   LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbHost
    lldbTarget

   LINK_COMPONENTS
    Support
  )
