lldb_tablegen(PlatformWasmProperties.inc -gen-lldb-property-defs
  SOURCE PlatformWasmProperties.td
  TARGET LLDBPluginPlatformWasmPropertiesGen)

lldb_tablegen(PlatformWasmPropertiesEnum.inc -gen-lldb-property-enum-defs
  SOURCE PlatformWasmProperties.td
  TARGET LLDBPluginPlatformWasmPropertiesEnumGen)

lldb_tablegen(PlatformWasmProperties.json -gen-lldb-property-docs-json
  SOURCE PlatformWasmProperties.td
  TARGET LLDBPluginPlatformWasmPropertiesJsonGen)

add_lldb_library(lldbPluginPlatformWasm PLUGIN
  PlatformWasm.cpp
  PlatformWasmRemoteGDBServer.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbTarget
    lldbUtility
  LINK_COMPONENTS
    Support
  )

add_dependencies(lldbPluginPlatformWasm
  LLDBPluginPlatformWasmPropertiesGen
  LLDBPluginPlatformWasmPropertiesEnumGen)
