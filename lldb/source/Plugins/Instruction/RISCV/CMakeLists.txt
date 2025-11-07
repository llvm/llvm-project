add_lldb_library(lldbPluginInstructionRISCV PLUGIN
  EmulateInstructionRISCV.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbCore
    lldbInterpreter
    lldbSymbol
    lldbPluginProcessUtility
  )
