add_lldb_library(lldbTreeSitter
  TreeSitterHighlighter.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbUtility
    ${TreeSitter_LIBRARY}
)
