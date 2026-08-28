add_lldb_library(lldbPluginInstrumentationRuntimeUtility
  Utility.cpp

  LINK_LIBS
    lldbCore
    lldbTarget
  )
