add_lldb_library(lldbPluginBugReporterGitHub PLUGIN
  BugReporterGitHub.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbCore
    lldbHost
  )
