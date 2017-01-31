add_lldb_library(lldbPluginABISysV_ppc PLUGIN
  ABISysV_ppc.cpp

  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbTarget
  LINK_COMPONENTS
    Support
  )
