add_lldb_library(lldbPluginABIAArch64 PLUGIN
  ABIMacOSX_arm64.cpp
  ABISysV_arm64.cpp

  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbTarget
  LINK_COMPONENTS
    Support
  )
