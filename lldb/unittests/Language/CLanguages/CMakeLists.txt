add_lldb_unittest(LanguageCLanguagesTests
  CLanguagesTest.cpp

  LINK_LIBS
    lldbPluginCPlusPlusLanguage
    lldbPluginObjCLanguage
    lldbPluginObjCPlusPlusLanguage
)
