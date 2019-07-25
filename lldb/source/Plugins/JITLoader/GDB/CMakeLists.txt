lldb_tablegen(Properties.inc -gen-lldb-property-defs
  SOURCE Properties.td
  TARGET LLDBPluginJITLoaderGDBPropertiesGen)

lldb_tablegen(PropertiesEnum.inc -gen-lldb-property-enum-defs
  SOURCE Properties.td
  TARGET LLDBPluginJITLoaderGDBPropertiesEnumGen)

add_lldb_library(lldbPluginJITLoaderGDB PLUGIN
  JITLoaderGDB.cpp

  LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbInterpreter
    lldbSymbol
    lldbTarget
    lldbUtility
  LINK_COMPONENTS
    Support
  )

add_dependencies(lldbPluginJITLoaderGDB
  LLDBPluginJITLoaderGDBPropertiesGen
  LLDBPluginJITLoaderGDBPropertiesEnumGen)
