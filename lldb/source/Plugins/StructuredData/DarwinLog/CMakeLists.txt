lldb_tablegen(Properties.inc -gen-lldb-property-defs
  SOURCE Properties.td
  TARGET LLDBPluginStructuredDataDarwinLogPropertiesGen)

lldb_tablegen(PropertiesEnum.inc -gen-lldb-property-enum-defs
  SOURCE Properties.td
  TARGET LLDBPluginStructuredDataDarwinLogPropertiesEnumGen)

add_lldb_library(lldbPluginStructuredDataDarwinLog PLUGIN
  StructuredDataDarwinLog.cpp

  LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbHost
    lldbInterpreter
    lldbTarget
  )

add_dependencies(lldbPluginStructuredDataDarwinLog
  LLDBPluginStructuredDataDarwinLogPropertiesGen
  LLDBPluginStructuredDataDarwinLogPropertiesEnumGen)
