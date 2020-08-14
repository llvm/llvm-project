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

if(APPLE_EMBEDDED)
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
      lldbHost
      lldbInitialization
      ${LLDB_PLUGINS}
      lldbPluginInstructionARM
      lldbPluginInstructionMIPS
      lldbPluginInstructionMIPS64
      ${LLDB_SYSTEM_LIBS}

    LINK_COMPONENTS
      Support
)

target_include_directories(lldb-server PRIVATE "${LLDB_SOURCE_DIR}/source")
target_link_libraries(lldb-server PRIVATE ${LLDB_SYSTEM_LIBS})
