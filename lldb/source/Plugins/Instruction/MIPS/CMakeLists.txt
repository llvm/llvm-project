add_lldb_library(lldbPluginInstructionMIPS PLUGIN
  EmulateInstructionMIPS.cpp

  LINK_LIBS
    lldbCore
    lldbInterpreter
    lldbSymbol
    lldbTarget
    lldbPluginProcessUtility
  LINK_COMPONENTS
    MC
    Support
  )
