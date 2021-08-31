add_lldb_library(lldbPluginTraceCommon
  ThreadPostMortemTrace.cpp
  TraceJSONStructs.cpp
  TraceSessionFileParser.cpp
  TraceSessionSaver.cpp

  LINK_LIBS
    lldbCore
    lldbTarget
  )
