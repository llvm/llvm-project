include_directories(../../Utility)
include_directories(../Common)

set(PROC_WINDOWS_MINIDUMP_SOURCES
  ProcessWinMiniDump.cpp
  ThreadWinMiniDump.cpp
  )

if (CMAKE_SIZEOF_VOID_P EQUAL 4)
  set(PROC_WINDOWS_MINIDUMP_SOURCES ${PROC_WINDOWS_MINIDUMP_SOURCES}
    x86/RegisterContextWindowsMiniDump_x86.cpp
    )
elseif (CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(PROC_WINDOWS_MINIDUMP_SOURCES ${PROC_WINDOWS_MINIDUMP_SOURCES}
    x64/RegisterContextWindowsMiniDump_x64.cpp
    )
endif()

add_lldb_library(lldbPluginProcessWinMiniDump
  ${PROC_WINDOWS_MINIDUMP_SOURCES}
  )
