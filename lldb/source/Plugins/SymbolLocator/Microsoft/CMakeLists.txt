lldb_tablegen(SymbolLocatorMicrosoftProperties.inc -gen-lldb-property-defs
  SOURCE SymbolLocatorMicrosoftProperties.td
  TARGET LLDBPluginSymbolLocatorMicrosoftPropertiesGen)

lldb_tablegen(SymbolLocatorMicrosoftPropertiesEnum.inc -gen-lldb-property-enum-defs
  SOURCE SymbolLocatorMicrosoftProperties.td
  TARGET LLDBPluginSymbolLocatorMicrosoftPropertiesEnumGen)

add_lldb_library(lldbPluginSymbolLocatorMicrosoft PLUGIN
  SymbolLocatorMicrosoft.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbSymbol
  )

add_dependencies(lldbPluginSymbolLocatorMicrosoft
  LLDBPluginSymbolLocatorMicrosoftPropertiesGen
  LLDBPluginSymbolLocatorMicrosoftPropertiesEnumGen)
