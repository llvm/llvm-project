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

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
  list(APPEND LLDB_PLUGINS lldbPluginObjectFileMachO)
elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
  list(APPEND LLDB_PLUGINS lldbPluginObjectFilePECOFF)
else()
  list(APPEND LLDB_PLUGINS lldbPluginObjectFileELF)
endif()

if(IOS)
  if(LLDB_CODESIGN_IDENTITY)
    # Use explicit LLDB identity
    set(LLVM_CODESIGNING_IDENTITY ${LLDB_CODESIGN_IDENTITY})
  else()
    # Use explicit LLVM identity or default to lldb_codesign if empty
    if(NOT LLVM_CODESIGNING_IDENTITY)
      set(LLVM_CODESIGNING_IDENTITY lldb_codesign)
    endif()
  endif()
endif()

add_lldb_tool(lldb-server
    Acceptor.cpp
    lldb-gdbserver.cpp
    lldb-platform.cpp
    lldb-server.cpp
    LLDBServerUtilities.cpp
    SystemInitializerLLGS.cpp

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

target_link_libraries(lldb-server PRIVATE ${LLDB_SYSTEM_LIBS})
