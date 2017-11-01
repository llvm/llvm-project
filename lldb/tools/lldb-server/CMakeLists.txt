if ( CMAKE_SYSTEM_NAME MATCHES "Linux" )
include_directories(
  ../../../../llvm/include
  ../../source/Plugins/Process/Linux
  ../../source/Plugins/Process/POSIX
  )
endif ()

if ( CMAKE_SYSTEM_NAME MATCHES "FreeBSD" )
include_directories(
  ../../../../llvm/include
  ../../source/Plugins/Process/FreeBSD
  ../../source/Plugins/Process/POSIX
  )
endif ()

if ( CMAKE_SYSTEM_NAME MATCHES "NetBSD" )
include_directories(
  ../../../../llvm/include
  ../../source/Plugins/Process/NetBSD
  ../../source/Plugins/Process/POSIX
  )
endif ()

include_directories(../../source)

set(LLDB_PLUGINS)

if(CMAKE_SYSTEM_NAME MATCHES "Linux|Android")
  list(APPEND LLDB_PLUGINS lldbPluginProcessLinux)
endif()

if(CMAKE_SYSTEM_NAME MATCHES "NetBSD")
  list(APPEND LLDB_PLUGINS lldbPluginProcessNetBSD)
endif()

add_lldb_tool(lldb-server INCLUDE_IN_FRAMEWORK
    Acceptor.cpp
    lldb-gdbserver.cpp
    lldb-platform.cpp
    lldb-server.cpp
    LLDBServerUtilities.cpp

    LINK_LIBS
      lldbBase
      lldbCore
      lldbHost
      lldbInitialization
      lldbInterpreter
      ${LLDB_PLUGINS}
      ${LLDB_SYSTEM_LIBS}

    LINK_COMPONENTS
      Support
)

target_link_libraries(lldb-server ${LLDB_SYSTEM_LIBS})
