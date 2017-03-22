add_lldb_library(lldbPluginStructuredDataDarwinLog PLUGIN
  StructuredDataDarwinLog.cpp

  LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbHost
    lldbInterpreter
    lldbTarget
  )
