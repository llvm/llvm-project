lldb_tablegen(SymbolLocatorSymStoreProperties.inc -gen-lldb-property-defs
  SOURCE SymbolLocatorSymStoreProperties.td
  TARGET LLDBPluginSymbolLocatorSymStorePropertiesGen)

lldb_tablegen(SymbolLocatorSymStorePropertiesEnum.inc -gen-lldb-property-enum-defs
  SOURCE SymbolLocatorSymStoreProperties.td
  TARGET LLDBPluginSymbolLocatorSymStorePropertiesEnumGen)

add_lldb_library(lldbPluginSymbolLocatorSymStore PLUGIN
  SymbolLocatorSymStore.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbSymbol

  LINK_COMPONENTS
    SupportHTTP
  )

add_dependencies(lldbPluginSymbolLocatorSymStore
  LLDBPluginSymbolLocatorSymStorePropertiesGen
  LLDBPluginSymbolLocatorSymStorePropertiesEnumGen)
