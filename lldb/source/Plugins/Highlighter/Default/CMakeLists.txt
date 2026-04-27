add_lldb_library(lldbPluginHighlighterDefault PLUGIN
  DefaultHighlighter.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbCore
    lldbUtility
)
