lldb_tablegen(StructuredDataDarwinLogProperties.inc -gen-lldb-property-defs
  SOURCE StructuredDataDarwinLogProperties.td
  TARGET LLDBPluginStructuredDataDarwinLogPropertiesGen)

lldb_tablegen(StructuredDataDarwinLogPropertiesEnum.inc -gen-lldb-property-enum-defs
  SOURCE StructuredDataDarwinLogProperties.td
  TARGET LLDBPluginStructuredDataDarwinLogPropertiesEnumGen)

lldb_tablegen(StructuredDataDarwinLogProperties.json -dump-json
  SOURCE StructuredDataDarwinLogProperties.td
  TARGET LLDBPluginStructuredDataDarwinLogPropertiesJsonGen)

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
