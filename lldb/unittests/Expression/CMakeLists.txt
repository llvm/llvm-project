add_lldb_unittest(ExpressionTests
  DWARFExpressionTest.cpp
  ClangParserTest.cpp
  SwiftParserTest.cpp

  LINK_LIBS
    lldbCore
    lldbPluginExpressionParserClang
    lldbUtility
    lldbUtilityHelpers
    LLVMTestingSupport
  )
