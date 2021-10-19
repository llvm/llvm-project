add_lldb_unittest(ExpressionTests
  ClangParserTest.cpp
  ClangExpressionDeclMapTest.cpp
  DiagnosticManagerTest.cpp
  DWARFExpressionTest.cpp
  CppModuleConfigurationTest.cpp

  LINK_LIBS
    lldbCore
    lldbPluginObjectFileELF
    lldbPluginPlatformLinux
    lldbPluginExpressionParserClang
    lldbPluginTypeSystemClang
    lldbUtility
    lldbUtilityHelpers
    lldbSymbolHelpers
    LLVMTestingSupport
  )
