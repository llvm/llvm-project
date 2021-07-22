add_lldb_library(lldbPluginTraceCommon
  ThreadPostMortemTrace.cpp
  TraceSessionFileParser.cpp

  LINK_LIBS
    lldbCore
    lldbTarget
  )
