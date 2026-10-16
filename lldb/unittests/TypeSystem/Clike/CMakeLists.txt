add_lldb_unittest(TypeSystemClikeUnitTests
  IdentifierTest.cpp
  LanguageOptsTest.cpp
  NamedTypeTest.cpp
  TypeTest.cpp

  LINK_LIBS
    lldbPluginTypeSystemClike
  )
