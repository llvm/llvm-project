if (NOT OPENCSD_INCLUDE_PATH)
  message (FATAL_ERROR "OpenCSD include path not provided")
endif()

if (NOT EXISTS "${OPENCSD_INCLUDE_PATH}")
  message (FATAL_ERROR "invalid OpenCSD include path provided")
endif()
include_directories(${OPENCSD_INCLUDE_PATH})

message(STATUS "Using OpenCSD include path: ${OPENCSD_INCLUDE_PATH}")

find_library(OPENCSD_LIBRARY opencsd PATHS ${OPENCSD_LIBRARY_PATH} REQUIRED)

add_lldb_library(lldbPluginTraceArmETM PLUGIN
  TraceArmETM.cpp
  TraceArmETMJSONStructs.cpp
  TraceArmETMBundleLoader.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbTarget
    lldbPluginTraceCommon
    ${OPENCSD_LIBRARY}
  )
