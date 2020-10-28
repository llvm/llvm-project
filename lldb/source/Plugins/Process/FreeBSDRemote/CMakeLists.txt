add_lldb_library(lldbPluginProcessFreeBSDRemote
  NativeProcessFreeBSD.cpp
  NativeRegisterContextFreeBSD.cpp
  NativeRegisterContextFreeBSD_x86_64.cpp
  NativeThreadFreeBSD.cpp

  LINK_LIBS
    lldbHost
    lldbSymbol
    lldbTarget
    lldbUtility
    lldbPluginProcessPOSIX
    lldbPluginProcessUtility
  LINK_COMPONENTS
    Support
  )
