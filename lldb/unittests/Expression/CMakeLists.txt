add_lldb_unittest(ExpressionTests
  ClangParserTest.cpp
  GoParserTest.cpp

  LINK_LIBS
    lldbCore
    lldbPluginExpressionParserGo
    lldbPluginExpressionParserClang
  )
