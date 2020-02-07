add_lldb_library(lldbPluginABIARM PLUGIN
  ABIMacOSX_arm.cpp
  ABISysV_arm.cpp

  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbTarget
    lldbPluginProcessUtility
  LINK_COMPONENTS
    Support
  )
