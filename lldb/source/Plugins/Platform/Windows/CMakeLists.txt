lldb_tablegen(PlatformWindowsProperties.inc -gen-lldb-property-defs
  SOURCE PlatformWindowsProperties.td
  TARGET LLDBPluginPlatformWindowsPropertiesGen)

lldb_tablegen(PlatformWindowsPropertiesEnum.inc -gen-lldb-property-enum-defs
  SOURCE PlatformWindowsProperties.td
  TARGET LLDBPluginPlatformWindowsPropertiesEnumGen)

lldb_tablegen(PlatformWindowsProperties.json -dump-json
  SOURCE PlatformWindowsProperties.td
  TARGET LLDBPluginPlatformWindowsPropertiesJsonGen)

add_lldb_library(lldbPluginPlatformWindows PLUGIN
  PlatformWindows.cpp

   LINK_COMPONENTS
    Support
   LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbHost
    lldbTarget
    lldbPluginPlatformGDB
  )

add_dependencies(lldbPluginPlatformWindows
  LLDBPluginPlatformWindowsPropertiesGen
  LLDBPluginPlatformWindowsPropertiesEnumGen)
