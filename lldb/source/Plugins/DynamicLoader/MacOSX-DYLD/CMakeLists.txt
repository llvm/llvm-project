lldb_tablegen(DynamicLoaderDarwinProperties.inc -gen-lldb-property-defs
  SOURCE DynamicLoaderDarwinProperties.td
  TARGET LLDBPluginDynamicLoaderDarwinPropertiesGen)

lldb_tablegen(DynamicLoaderDarwinPropertiesEnum.inc -gen-lldb-property-enum-defs
  SOURCE DynamicLoaderDarwinProperties.td
  TARGET LLDBPluginDynamicLoaderDarwinPropertiesEnumGen)

add_lldb_library(lldbPluginDynamicLoaderMacOSXDYLD PLUGIN
  DynamicLoaderMacOSXDYLD.cpp
  DynamicLoaderMacOS.cpp
  DynamicLoaderDarwin.cpp
  DynamicLoaderDarwinProperties.cpp

  LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbExpression
    lldbHost
    lldbSymbol
    lldbTarget
    lldbUtility
    lldbPluginTypeSystemClang
  LINK_COMPONENTS
    Support
    TargetParser
  )

add_dependencies(lldbPluginDynamicLoaderMacOSXDYLD
  LLDBPluginDynamicLoaderDarwinPropertiesGen
  LLDBPluginDynamicLoaderDarwinPropertiesEnumGen)
