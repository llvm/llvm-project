add_lldb_library(lldbPluginHighlighterTreeSitterRust PLUGIN
  RustTreeSitterHighlighter.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbCore
    lldbUtility
    lldbTreeSitter
    tree-sitter-rust
)

add_tree_sitter_grammar(tree-sitter-rust
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)
