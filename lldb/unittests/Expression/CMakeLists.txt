add_lldb_unittest(ExpressionTests
  ClangParserTest.cpp

  LINK_LIBS
    lldbCore
    lldbPluginExpressionParserClang
    lldbUtility
    lldbUtilityHelpers
  )
