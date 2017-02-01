add_lldb_unittest(ExpressionTests
  GoParserTest.cpp

  LINK_LIBS
    lldbCore
    lldbPluginExpressionParserGo
  )
