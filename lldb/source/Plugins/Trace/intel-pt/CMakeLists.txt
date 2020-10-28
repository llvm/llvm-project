if (NOT LIBIPT_INCLUDE_PATH)
  message (FATAL_ERROR "libipt include path not provided")
endif()

if (NOT EXISTS "${LIBIPT_INCLUDE_PATH}")
  message (FATAL_ERROR "invalid libipt include path provided")
endif()
include_directories(${LIBIPT_INCLUDE_PATH})

find_library(LIBIPT_LIBRARY ipt PATHS ${LIBIPT_LIBRARY_PATH} REQUIRED)

add_lldb_library(lldbPluginTraceIntelPT PLUGIN
  TraceIntelPT.cpp
  TraceIntelPTSettingsParser.cpp

  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbTarget
    ${LIBIPT_LIBRARY}
  LINK_COMPONENTS
    Support
  )
