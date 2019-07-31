
add_lldb_library(lldbPluginProcessWindowsCommon PLUGIN
  DebuggerThread.cpp
  LocalDebugDelegate.cpp
  ProcessDebugger.cpp
  ProcessWindows.cpp
  ProcessWindowsLog.cpp
  RegisterContextWindows.cpp
  TargetThreadWindows.cpp
  x64/RegisterContextWindows_x64.cpp
  x86/RegisterContextWindows_x86.cpp
  # TODO add support for ARM (NT) and ARM64

  LINK_LIBS
    lldbCore
    lldbHost
    lldbInterpreter
    lldbPluginDynamicLoaderWindowsDYLD
    lldbSymbol
    lldbTarget
    ws2_32
    rpcrt4
  LINK_COMPONENTS
    Support
  )
