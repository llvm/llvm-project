set(UPMEM_INC $ENV{UPMEM_HOME}/usr/share/upmem/include/host)
include_directories(${UPMEM_INC})

add_lldb_library(lldbPluginProcessDpu PLUGIN
  DpuRank.cpp
  ProcessDpu.cpp
  RegisterContextDpu.cpp
  RegisterInfo_dpu.cpp
  ThreadDpu.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbSymbol
    lldbTarget
    lldbUtility
    lldbPluginProcessPOSIX
    lldbPluginProcessUtility
    dpucni
  LINK_COMPONENTS
    Support
  EXTRA_CXXFLAGS -g
  )
