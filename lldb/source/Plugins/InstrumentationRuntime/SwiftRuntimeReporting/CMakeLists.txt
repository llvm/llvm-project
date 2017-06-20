add_lldb_library(lldbPluginInstrumentationRuntimeSwiftRuntimeReporting PLUGIN
  SwiftRuntimeReporting.cpp

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
