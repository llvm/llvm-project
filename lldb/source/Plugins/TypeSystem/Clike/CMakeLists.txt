add_lldb_library(lldbPluginTypeSystemClike PLUGIN
  Identifier.cpp
  TypeSystemClike.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbUtility
)
