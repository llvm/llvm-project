add_lldb_library(lldbPluginABISysV_arm64 PLUGIN
  ABISysV_arm64.cpp

  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbTarget
  LINK_COMPONENTS
    Support
  )
