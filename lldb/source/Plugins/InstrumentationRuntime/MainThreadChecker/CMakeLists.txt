add_lldb_library(lldbPluginInstrumentationRuntimeMainThreadChecker PLUGIN
  InstrumentationRuntimeMainThreadChecker.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbExpression
    lldbInterpreter
    lldbSymbol
    lldbTarget
  )
