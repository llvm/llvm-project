add_lldb_library(lldbPluginDynamicLoaderMacOSXDYLD PLUGIN
  DynamicLoaderMacOSXDYLD.cpp
  DynamicLoaderMacOS.cpp
  DynamicLoaderDarwin.cpp

  LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbExpression
    lldbHost
    lldbSymbol
    lldbTarget
    lldbUtility
  LINK_COMPONENTS
    Support
  )
