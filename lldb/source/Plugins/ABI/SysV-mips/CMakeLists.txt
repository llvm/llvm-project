add_lldb_library(lldbPluginABISysV_mips PLUGIN
  ABISysV_mips.cpp

  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbTarget
  LINK_COMPONENTS
    Support
  )
