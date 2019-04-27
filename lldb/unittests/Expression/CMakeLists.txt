add_lldb_unittest(ExpressionTests
  ClangParserTest.cpp
  SwiftParserTest.cpp

  LINK_LIBS
    lldbCore
    lldbPluginExpressionParserClang
    lldbUtility
    lldbUtilityHelpers
  )
