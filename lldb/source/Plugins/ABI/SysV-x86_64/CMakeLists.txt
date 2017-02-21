add_lldb_library(lldbPluginABISysV_x86_64 PLUGIN
  ABISysV_x86_64.cpp

  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbTarget
  LINK_COMPONENTS
    Support
  )
