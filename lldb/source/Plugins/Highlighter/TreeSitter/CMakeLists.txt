add_lldb_library(lldbTreeSitter
  TreeSitterHighlighter.cpp

  LINK_COMPONENTS
    Support
    ADT
  LINK_LIBS
    lldbUtility
    ${TreeSitter_LIBRARY}
)
