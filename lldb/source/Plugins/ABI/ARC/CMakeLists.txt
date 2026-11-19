add_lldb_library(lldbPluginABIARC PLUGIN
  ABISysV_arc.cpp

  LINK_COMPONENTS
    Support
    TargetParser
  LINK_LIBS
    lldbCore
    lldbRegister
    lldbSymbol
    lldbTarget
    lldbPluginProcessUtility
    lldbValueObject
  )
