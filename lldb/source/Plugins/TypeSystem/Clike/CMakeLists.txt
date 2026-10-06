add_lldb_library(lldbPluginTypeSystemClike PLUGIN
  TypeSystemClike.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbUtility
)
