add_lldb_library(lldbPluginBugReporterNone PLUGIN
  BugReporterNone.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbCore
  )
