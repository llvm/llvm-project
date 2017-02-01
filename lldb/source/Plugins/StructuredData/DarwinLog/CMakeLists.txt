add_lldb_library(lldbPluginStructuredDataDarwinLog PLUGIN
  StructuredDataDarwinLog.cpp

  LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbInterpreter
    lldbTarget
  )
