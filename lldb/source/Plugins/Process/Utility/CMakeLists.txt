# TODO: Clean up this directory and its dependencies
set_property(DIRECTORY PROPERTY LLDB_PLUGIN_KIND ProcessUtility)

add_lldb_library(lldbPluginProcessUtility
  AuxVector.cpp
  FreeBSDSignals.cpp
  GDBRemoteSignals.cpp
  HistoryThread.cpp
  HistoryUnwind.cpp
  InferiorCallPOSIX.cpp
  LinuxProcMaps.cpp
  LinuxSignals.cpp
  MemoryTagManagerAArch64MTE.cpp
  NativeProcessSoftwareSingleStep.cpp
  NetBSDSignals.cpp
  OpenBSDSignals.cpp
  RegisterContextDummy.cpp
  RegisterContextHistory.cpp
  RegisterContextMemory.cpp
  RegisterContextThreadMemory.cpp
  StopInfoMachException.cpp
  ThreadMemory.cpp

  LINK_COMPONENTS
    Support
    TargetParser
  LINK_LIBS
    lldbRegister
    lldbBreakpoint
    lldbCore
    lldbDataFormatters
    lldbExpression
    lldbHost
    lldbSymbol
    lldbTarget
    lldbUtility
    lldbValueObject
  )
