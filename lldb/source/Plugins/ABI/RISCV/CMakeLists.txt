add_lldb_library(lldbPluginABIRISCV PLUGIN
  ABISysV_riscv.cpp

  LINK_COMPONENTS
    Support
    TargetParser
  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbTarget
    lldbPluginProcessUtility
    lldbValueObject
  )
