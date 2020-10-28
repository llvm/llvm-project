add_lldb_unittest(ExpressionTests
  ClangParserTest.cpp
  ClangExpressionDeclMapTest.cpp
  DiagnosticManagerTest.cpp
  DWARFExpressionTest.cpp
  CppModuleConfigurationTest.cpp

  LINK_LIBS
    lldbCore
    lldbPluginExpressionParserClang
    lldbPluginTypeSystemClang
    lldbUtility
    lldbUtilityHelpers
    lldbSymbolHelpers
    LLVMTestingSupport
  )
