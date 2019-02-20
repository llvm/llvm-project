if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
  include_directories(${LIBXML2_INCLUDE_DIR})
elseif (CMAKE_SYSTEM_NAME MATCHES "Linux")
  list(APPEND LLDB_PLUGINS lldbPluginProcessLinux)
endif()

set(LLDB_PLUGINS
  lldbPluginProcessUtility
  lldbPluginPlatformMacOSX
)

if(HAVE_LIBCOMPRESSION)
  set(LIBCOMPRESSION compression)
endif()

add_lldb_library(lldbPluginProcessGDBRemote PLUGIN
  GDBRemoteClientBase.cpp
  GDBRemoteCommunication.cpp
  GDBRemoteCommunicationClient.cpp
  GDBRemoteCommunicationHistory.cpp
  GDBRemoteCommunicationReplayServer.cpp
  GDBRemoteCommunicationServer.cpp
  GDBRemoteCommunicationServerCommon.cpp
  GDBRemoteCommunicationServerLLGS.cpp
  GDBRemoteCommunicationServerPlatform.cpp
  GDBRemoteRegisterContext.cpp
  ProcessGDBRemote.cpp
  ProcessGDBRemoteLog.cpp
  ThreadGDBRemote.cpp

  LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbDataFormatters
    lldbHost
    lldbInterpreter
    lldbSymbol
    lldbTarget
    lldbUtility
    ${LLDB_PLUGINS}
    ${LIBCOMPRESSION}
  LINK_COMPONENTS
    Support
  )
