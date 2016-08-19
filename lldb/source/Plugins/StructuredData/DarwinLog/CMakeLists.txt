list(APPEND SOURCES
  StructuredDataDarwinLog.cpp
  )

add_lldb_library(lldbPluginStructuredDataDarwinLog ${SOURCES})
