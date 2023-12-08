add_lldb_unittest(HighlighterTests
  HighlighterTest.cpp

  LINK_LIBS
    lldbPluginCPlusPlusLanguage
    lldbPluginObjCLanguage
    lldbPluginObjCPlusPlusLanguage
  )
