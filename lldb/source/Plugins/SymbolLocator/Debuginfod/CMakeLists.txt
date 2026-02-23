lldb_tablegen(SymbolLocatorDebuginfodProperties.inc -gen-lldb-property-defs
  SOURCE SymbolLocatorDebuginfodProperties.td
  TARGET LLDBPluginSymbolLocatorDebuginfodPropertiesGen)

lldb_tablegen(SymbolLocatorDebuginfodPropertiesEnum.inc -gen-lldb-property-enum-defs
  SOURCE SymbolLocatorDebuginfodProperties.td
  TARGET LLDBPluginSymbolLocatorDebuginfodPropertiesEnumGen)

lldb_tablegen(SymbolLocatorDebuginfodProperties.json -dump-json
  SOURCE SymbolLocatorDebuginfodProperties.td
  TARGET LLDBPluginSymbolLocatorDebuginfodPropertiesJsonGen)

add_lldb_library(lldbPluginSymbolLocatorDebuginfod PLUGIN
  SymbolLocatorDebuginfod.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbSymbol
    LLVMDebuginfod
  )

add_dependencies(lldbPluginSymbolLocatorDebuginfod
  LLDBPluginSymbolLocatorDebuginfodPropertiesGen
  LLDBPluginSymbolLocatorDebuginfodPropertiesEnumGen)
