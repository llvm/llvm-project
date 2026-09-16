if (LLDB_ENABLE_TREESITTER)
  set(TREESITTER_HIGHLIGHTER_PLUGINS
    lldbPluginHighlighterTreeSitterRust
    lldbPluginHighlighterTreeSitterSwift
  )
endif()

add_lldb_unittest(HighlighterTests
  HighlighterTest.cpp

  LINK_LIBS
    lldbPluginHighlighterDefault
    lldbPluginHighlighterClang
    lldbPluginCPlusPlusLanguage
    lldbPluginObjCLanguage
    lldbPluginObjCPlusPlusLanguage
    ${TREESITTER_HIGHLIGHTER_PLUGINS}
)
