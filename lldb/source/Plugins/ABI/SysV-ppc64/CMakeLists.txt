add_lldb_library(lldbPluginABISysV_ppc64 PLUGIN
  ABISysV_ppc64.cpp

  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbTarget
  LINK_COMPONENTS
    Support
  )
