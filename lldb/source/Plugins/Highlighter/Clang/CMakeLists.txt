add_lldb_library(lldbPluginHighlighterClang PLUGIN
  ClangHighlighter.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbCore
    lldbUtility
)
