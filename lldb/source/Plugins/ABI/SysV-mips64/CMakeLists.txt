add_lldb_library(lldbPluginABISysV_mips64 PLUGIN
  ABISysV_mips64.cpp

  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbTarget
  LINK_COMPONENTS
    Support
  )
