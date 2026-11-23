add_lldb_library(lldbPluginInstructionLoongArch PLUGIN
  EmulateInstructionLoongArch.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbCore
    lldbInterpreter
    lldbPluginProcessUtility
    lldbSymbol
  )
