add_lldb_library(lldbPluginInstrumentationRuntimeUtility
  ReportRetriever.cpp
  Utility.cpp

  LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbExpression
    lldbSymbol
    lldbTarget
    lldbValueObject
  )
