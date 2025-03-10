add_lldb_library(lldbPluginInstrumentationRuntimeUBSan PLUGIN
  InstrumentationRuntimeUBSan.cpp

  LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbExpression
    lldbInterpreter
    lldbSymbol
    lldbTarget
    lldbValueObject
  LINK_COMPONENTS
    Support
  )
