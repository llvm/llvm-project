add_lldb_library(lldbPluginInstrumentationRuntimeMainThreadChecker PLUGIN
  MainThreadCheckerRuntime.cpp

  LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbExpression
    lldbInterpreter
    lldbSymbol
    lldbTarget
  LINK_COMPONENTS
    Support
  )
