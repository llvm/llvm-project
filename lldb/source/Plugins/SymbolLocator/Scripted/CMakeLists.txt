set_property(DIRECTORY PROPERTY LLDB_PLUGIN_KIND SymbolLocator)

add_lldb_library(lldbPluginSymbolLocatorScripted PLUGIN
  SymbolLocatorScripted.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbInterpreter
    lldbSymbol
    lldbTarget
    lldbUtility
  )
