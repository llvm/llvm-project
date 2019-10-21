add_lldb_library(lldbPluginABISysV_arc PLUGIN
  ABISysV_arc.cpp

  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbTarget
    lldbPluginProcessUtility
  LINK_COMPONENTS
    Support
  )
