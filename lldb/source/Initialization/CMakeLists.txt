if ( CMAKE_SYSTEM_NAME MATCHES "Darwin" )
  list(APPEND EXTRA_PLUGINS lldbPluginObjectFileMachO)
endif()

if ( CMAKE_SYSTEM_NAME MATCHES "Linux|Android|FreeBSD|NetBSD" )
  list(APPEND EXTRA_PLUGINS lldbPluginProcessPOSIX)
endif()

if ( CMAKE_SYSTEM_NAME MATCHES "Windows" )
  list(APPEND LLDB_USED_LIBS
    lldbPluginProcessWindowsCommon
    ws2_32
    rpcrt4
    )
endif ()

add_lldb_library(lldbInitialization
  SystemInitializerCommon.cpp
  SystemInitializer.cpp
  SystemLifetimeManager.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbPluginInstructionARM
    lldbPluginInstructionMIPS
    lldbPluginInstructionMIPS64
    lldbPluginObjectContainerBSDArchive
    lldbPluginObjectContainerMachOArchive
    lldbPluginObjectFileELF
    lldbPluginObjectFilePECOFF
    lldbPluginProcessGDBRemote
    ${EXTRA_PLUGINS}
  LINK_COMPONENTS
    Support
  )
