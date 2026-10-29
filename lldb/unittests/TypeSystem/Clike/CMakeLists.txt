add_lldb_unittest(TypeSystemClikeUnitTests
  IdentifierTest.cpp
  LanguageOptsTest.cpp
  TypeTest.cpp

  LINK_LIBS
    lldbPluginTypeSystemClike
  )
