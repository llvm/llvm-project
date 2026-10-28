add_lldb_library(lldbPluginProcessOpenBSD
  NativeProcessOpenBSD.cpp
  NativeRegisterContextOpenBSD.cpp
  NativeRegisterContextOpenBSD_arch.cpp
  NativeThreadOpenBSD.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbHost
    lldbSymbol
    lldbTarget
    lldbUtility
    lldbPluginProcessPOSIX
    lldbPluginProcessUtility
  )
