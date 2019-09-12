add_lldb_unittest(ExpressionTests
  ClangParserTest.cpp
  DiagnosticManagerTest.cpp
  DWARFExpressionTest.cpp
  SwiftParserTest.cpp

  LINK_LIBS
    lldbCore
    lldbPluginExpressionParserClang
    lldbUtility
    lldbUtilityHelpers
    LLVMTestingSupport
  )
