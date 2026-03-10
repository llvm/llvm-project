add_lldb_library(lldbPluginTraceCommon
  ThreadPostMortemTrace.cpp
  TraceBundleLoader.cpp
  TraceJSONStructs.cpp

  LINK_LIBS
    lldbCore
    lldbTarget
  )
