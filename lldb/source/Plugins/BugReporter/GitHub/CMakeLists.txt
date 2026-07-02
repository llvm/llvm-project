add_lldb_library(lldbPluginGitHubReporter PLUGIN
  GitHubReporter.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbCore
    lldbHost
  )
