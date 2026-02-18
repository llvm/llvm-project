if (LLDB_ENABLE_TREESITTER)
  set(SWIFT_HIGHLIGHTER_PLUGIN lldbPluginHighlighterTreeSitterSwift)
endif()

add_lldb_unittest(HighlighterTests
  HighlighterTest.cpp

  LINK_LIBS
    lldbPluginHighlighterDefault
    lldbPluginHighlighterClang
    lldbPluginCPlusPlusLanguage
    lldbPluginObjCLanguage
    lldbPluginObjCPlusPlusLanguage
    ${SWIFT_HIGHLIGHTER_PLUGIN}
)
