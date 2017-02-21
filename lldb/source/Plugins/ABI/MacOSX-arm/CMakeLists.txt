add_lldb_library(lldbPluginABIMacOSX_arm PLUGIN
  ABIMacOSX_arm.cpp

  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbTarget
    lldbPluginProcessUtility
  LINK_COMPONENTS
    Support
  )
