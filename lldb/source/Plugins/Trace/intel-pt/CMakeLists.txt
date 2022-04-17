if (NOT LIBIPT_INCLUDE_PATH)
  message (FATAL_ERROR "libipt include path not provided")
endif()

if (NOT EXISTS "${LIBIPT_INCLUDE_PATH}")
  message (FATAL_ERROR "invalid libipt include path provided")
endif()
include_directories(${LIBIPT_INCLUDE_PATH})

find_library(LIBIPT_LIBRARY ipt PATHS ${LIBIPT_LIBRARY_PATH} REQUIRED)

lldb_tablegen(TraceIntelPTCommandOptions.inc -gen-lldb-option-defs
  SOURCE TraceIntelPTOptions.td
  TARGET TraceIntelPTOptionsGen)

add_lldb_library(lldbPluginTraceIntelPT PLUGIN
  CommandObjectTraceStartIntelPT.cpp
  DecodedThread.cpp
  TaskTimer.cpp
  LibiptDecoder.cpp
  ThreadDecoder.cpp
  TraceCursorIntelPT.cpp
  TraceIntelPT.cpp
  TraceIntelPTJSONStructs.cpp
  TraceIntelPTSessionFileParser.cpp
  TraceIntelPTSessionSaver.cpp

  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbTarget
    lldbPluginTraceCommon
    ${LIBIPT_LIBRARY}
  LINK_COMPONENTS
    Support
  )


add_dependencies(lldbPluginTraceIntelPT TraceIntelPTOptionsGen)
