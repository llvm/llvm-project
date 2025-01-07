add_lldb_library(lldbPluginABIARC PLUGIN
  ABISysV_arc.cpp

  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbTarget
    lldbPluginProcessUtility
    lldbValueObject
  LINK_COMPONENTS
    Support
    TargetParser
  )
