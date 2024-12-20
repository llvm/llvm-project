add_lldb_library(lldbPluginABILoongArch PLUGIN
  ABISysV_loongarch.cpp

  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbTarget
    lldbPluginProcessUtility
  LINK_COMPONENTS
    Support
    TargetParser
  )
