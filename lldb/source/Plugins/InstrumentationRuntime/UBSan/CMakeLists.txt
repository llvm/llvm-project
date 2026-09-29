add_lldb_library(lldbPluginInstrumentationRuntimeUBSan PLUGIN
  InstrumentationRuntimeUBSan.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbExpression
    lldbInterpreter
    lldbSymbol
    lldbTarget
    lldbValueObject
  )
