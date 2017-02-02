add_lldb_unittest(SymbolTests
  TestClangASTContext.cpp
  TestType.cpp

  LINK_LIBS
    lldbHost
    lldbSymbol
  )
