add_lldb_library(lldbPluginABILoongArch PLUGIN
  ABISysV_loongarch.cpp

  LINK_COMPONENTS
    Support
    TargetParser
  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbTarget
    lldbPluginProcessUtility
  )
