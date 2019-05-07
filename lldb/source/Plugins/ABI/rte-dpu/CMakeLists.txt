add_lldb_library(lldbPluginABIrte_dpu PLUGIN
  ABIrte_dpu.cpp

  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbTarget
  LINK_COMPONENTS
    Support
  )
