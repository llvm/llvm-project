add_lldb_library(lldbPluginObjectFileBreakpad PLUGIN
  ObjectFileBreakpad.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbSymbol
    lldbUtility
  LINK_COMPONENTS
    Support
  )
