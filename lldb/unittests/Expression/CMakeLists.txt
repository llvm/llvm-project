add_lldb_unittest(ExpressionTests
  DiagnosticManagerTest.cpp
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
