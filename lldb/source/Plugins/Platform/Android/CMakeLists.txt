lldb_tablegen(PlatformAndroidProperties.inc -gen-lldb-property-defs
  SOURCE PlatformAndroidProperties.td
  TARGET LLDBPluginPlatformAndroidPropertiesGen)

lldb_tablegen(PlatformAndroidPropertiesEnum.inc -gen-lldb-property-enum-defs
  SOURCE PlatformAndroidProperties.td
  TARGET LLDBPluginPlatformAndroidPropertiesEnumGen)

add_lldb_library(lldbPluginPlatformAndroid PLUGIN
  AdbClient.cpp
  PlatformAndroid.cpp
  PlatformAndroidRemoteGDBServer.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbCore
    lldbHost
    lldbValueObject
    lldbPluginPlatformLinux
    lldbPluginPlatformGDB
  )

add_dependencies(lldbPluginPlatformAndroid
  LLDBPluginPlatformAndroidPropertiesGen
  LLDBPluginPlatformAndroidPropertiesEnumGen)
