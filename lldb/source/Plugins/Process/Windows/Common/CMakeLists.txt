
add_lldb_library(lldbPluginProcessWindowsCommon PLUGIN
  DebuggerThread.cpp
  LocalDebugDelegate.cpp
  ProcessDebugger.cpp
  ProcessWindows.cpp
  ProcessWindowsLog.cpp
  RegisterContextWindows.cpp
  TargetThreadWindows.cpp

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

# TODO add support for ARM (NT) and ARM64
# TODO build these unconditionally as we cannot do cross-debugging or WoW
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  target_sources(lldbPluginProcessWindowsCommon PRIVATE
    x64/RegisterContextWindows_x64.cpp)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "i?86|X86")
  target_sources(lldbPluginProcessWindowsCommon PRIVATE
    x86/RegisterContextWindows_x86.cpp)
endif()
