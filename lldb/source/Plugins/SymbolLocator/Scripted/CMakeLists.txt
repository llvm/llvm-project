lldb_tablegen(SymbolLocatorScriptedProperties.inc -gen-lldb-property-defs
  SOURCE SymbolLocatorScriptedProperties.td
  TARGET LLDBPluginSymbolLocatorScriptedPropertiesGen)

lldb_tablegen(SymbolLocatorScriptedPropertiesEnum.inc -gen-lldb-property-enum-defs
  SOURCE SymbolLocatorScriptedProperties.td
  TARGET LLDBPluginSymbolLocatorScriptedPropertiesEnumGen)

set_property(DIRECTORY PROPERTY LLDB_PLUGIN_KIND SymbolLocator)

add_lldb_library(lldbPluginSymbolLocatorScripted PLUGIN
  SymbolLocatorScripted.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbInterpreter
    lldbSymbol
    lldbUtility
  )

add_dependencies(lldbPluginSymbolLocatorScripted
  LLDBPluginSymbolLocatorScriptedPropertiesGen
  LLDBPluginSymbolLocatorScriptedPropertiesEnumGen)
