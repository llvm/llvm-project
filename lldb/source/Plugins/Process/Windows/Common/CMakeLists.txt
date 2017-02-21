include_directories(.)
include_directories(../../Utility)

set(PROC_WINDOWS_COMMON_SOURCES
  DebuggerThread.cpp
  LocalDebugDelegate.cpp
  ProcessWindows.cpp
  ProcessWindowsLog.cpp
  RegisterContextWindows.cpp
  TargetThreadWindows.cpp
  )

if (CMAKE_SIZEOF_VOID_P EQUAL 4)
  set(PROC_WINDOWS_COMMON_SOURCES ${PROC_WINDOWS_COMMON_SOURCES}
    x86/RegisterContextWindows_x86.cpp
    )
elseif (CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(PROC_WINDOWS_COMMON_SOURCES ${PROC_WINDOWS_COMMON_SOURCES}
    x64/RegisterContextWindows_x64.cpp
    )
endif()

add_lldb_library(lldbPluginProcessWindowsCommon PLUGIN
  ${PROC_WINDOWS_COMMON_SOURCES}

  LINK_LIBS
    lldbCore
    lldbHost
    lldbInterpreter
    lldbTarget
    ws2_32
    rpcrt4
  LINK_COMPONENTS
    Support
  )
