add_lldb_library(lldbTreeSitter
  TreeSitterHighlighter.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbCore
    lldbUtility
    ${TreeSitter_LIBRARY}
)
