lldb_tablegen(PlatformGDBStubLauncherProperties.inc -gen-lldb-property-defs
  SOURCE PlatformGDBStubLauncherProperties.td
  TARGET LLDBPluginPlatformGDBStubLauncherPropertiesGen)

lldb_tablegen(PlatformGDBStubLauncherPropertiesEnum.inc -gen-lldb-property-enum-defs
  SOURCE PlatformGDBStubLauncherProperties.td
  TARGET LLDBPluginPlatformGDBStubLauncherPropertiesEnumGen)

lldb_tablegen(PlatformGDBStubLauncherProperties.json -dump-json
  SOURCE PlatformGDBStubLauncherProperties.td
  TARGET LLDBPluginPlatformGDBStubLauncherPropertiesJsonGen)

add_lldb_library(lldbPluginPlatformGDBStubLauncher PLUGIN
  PlatformGDBStubLauncher.cpp

  LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbHost
    lldbInterpreter
    lldbTarget
    lldbUtility
    lldbPluginProcessGDBRemote
  LINK_COMPONENTS
    Support
  )

add_dependencies(lldbPluginPlatformGDBStubLauncher
  LLDBPluginPlatformGDBStubLauncherPropertiesGen
  LLDBPluginPlatformGDBStubLauncherPropertiesEnumGen)
