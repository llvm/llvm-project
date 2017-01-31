add_lldb_library(lldbPluginABISysV_arm PLUGIN
  ABISysV_arm.cpp

  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbTarget
    lldbPluginProcessUtility
  LINK_COMPONENTS
    Support
  )
