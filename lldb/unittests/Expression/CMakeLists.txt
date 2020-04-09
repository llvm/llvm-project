add_lldb_unittest(ExpressionTests
  ClangParserTest.cpp
  ClangExpressionDeclMapTest.cpp
  DiagnosticManagerTest.cpp
  DWARFExpressionTest.cpp
  SwiftParserTest.cpp
  CppModuleConfigurationTest.cpp

  LINK_LIBS
    lldbCore
    lldbPluginExpressionParserClang
    lldbPluginTypeSystemClang
    lldbUtility
    lldbUtilityHelpers
    LLVMTestingSupport
    lldbPluginPlatformMacOSX
  )
