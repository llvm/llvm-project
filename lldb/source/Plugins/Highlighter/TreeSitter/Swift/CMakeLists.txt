add_lldb_library(lldbPluginHighlighterTreeSitterSwift PLUGIN
  SwiftTreeSitterHighlighter.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbCore
    lldbUtility
    lldbTreeSitter
    tree-sitter-swift
)

add_tree_sitter_grammar(tree-sitter-swift
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)
