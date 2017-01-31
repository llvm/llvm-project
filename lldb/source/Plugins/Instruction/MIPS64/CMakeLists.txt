add_lldb_library(lldbPluginInstructionMIPS64 PLUGIN
  EmulateInstructionMIPS64.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbInterpreter
    lldbSymbol
    lldbPluginProcessUtility
  LINK_COMPONENTS
    MC
    Support
  )
