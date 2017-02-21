add_lldb_library(lldbPluginABIMacOSX_arm64 PLUGIN
  ABIMacOSX_arm64.cpp
  
  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbTarget
  LINK_COMPONENTS
    Support
  )
