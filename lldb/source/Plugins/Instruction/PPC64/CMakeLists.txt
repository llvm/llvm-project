add_lldb_library(lldbPluginInstructionPPC64 PLUGIN
  EmulateInstructionPPC64.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbCore
    lldbInterpreter
    lldbSymbol
    lldbPluginProcessUtility
  )
