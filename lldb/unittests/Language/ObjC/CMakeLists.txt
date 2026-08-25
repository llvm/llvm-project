add_lldb_unittest(LanguageObjCTests
  GNUstepFormattersTest.cpp
  ObjCLanguageTest.cpp

  LINK_LIBS
    lldbPluginObjCLanguage
)
