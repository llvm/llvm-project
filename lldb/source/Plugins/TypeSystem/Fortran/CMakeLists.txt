add_lldb_library(lldbPluginTypeSystemFortran PLUGIN
  TypeSystemFortran.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbTarget
    lldbUtility
    lldbPluginSymbolFileDWARF
)
