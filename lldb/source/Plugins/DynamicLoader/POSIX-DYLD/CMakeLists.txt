lldb_tablegen(DynamicLoaderPOSIXDYLDProperties.inc -gen-lldb-property-defs
  SOURCE DynamicLoaderPOSIXDYLDProperties.td
  TARGET LLDBPluginDynamicLoaderPOSIXDYLDPropertiesGen)

lldb_tablegen(DynamicLoaderPOSIXDYLDPropertiesEnum.inc -gen-lldb-property-enum-defs
  SOURCE DynamicLoaderPOSIXDYLDProperties.td
  TARGET LLDBPluginDynamicLoaderPOSIXDYLDPropertiesEnumGen)

add_lldb_library(lldbPluginDynamicLoaderPosixDYLD PLUGIN
  DYLDRendezvous.cpp
  DynamicLoaderPOSIXDYLD.cpp

  LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbHost
    lldbSymbol
    lldbTarget
    lldbPluginProcessElfCore
    lldbPluginProcessUtility
  LINK_COMPONENTS
    Support
  )

add_dependencies(lldbPluginDynamicLoaderPosixDYLD
  LLDBPluginDynamicLoaderPOSIXDYLDPropertiesGen
  LLDBPluginDynamicLoaderPOSIXDYLDPropertiesEnumGen)
