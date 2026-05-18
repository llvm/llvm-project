add_lldb_library(lldbPluginSymbolFileBreakpad PLUGIN
  SymbolFileBreakpad.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbCore
    lldbHost
    lldbSymbol
    lldbUtility
    lldbPluginObjectFileBreakpad
  )
