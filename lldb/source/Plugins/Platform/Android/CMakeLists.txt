add_lldb_library(lldbPluginPlatformAndroid PLUGIN
  AdbClient.cpp
  PlatformAndroid.cpp
  PlatformAndroidRemoteGDBServer.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbPluginPlatformLinux
    lldbPluginPlatformGDB
  LINK_COMPONENTS
    Support
  )
