add_lldb_library(lldbPluginClangCommon PLUGIN
  ClangHighlighter.cpp

  LINK_LIBS
    lldbCore
    lldbUtility
  LINK_COMPONENTS
    Support
)
