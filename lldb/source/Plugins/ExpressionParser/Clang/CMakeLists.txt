if(NOT LLDB_BUILT_STANDALONE)
  set(tablegen_deps intrinsics_gen)
endif()

add_lldb_library(lldbPluginExpressionParserClang PLUGIN
  ASTResultSynthesizer.cpp
  ASTStructExtractor.cpp
  ASTUtils.cpp
  ClangASTSource.cpp
  ClangDeclVendor.cpp
  ClangExpressionDeclMap.cpp
  ClangExpressionParser.cpp
  ClangExpressionSourceCode.cpp
  ClangExpressionVariable.cpp
  ClangFunctionCaller.cpp
  ClangHost.cpp
  ClangModulesDeclVendor.cpp
  ClangPersistentVariables.cpp
  ClangUserExpression.cpp
  ClangUtilityFunction.cpp
  CppModuleConfiguration.cpp
  IRForTarget.cpp
  IRDynamicChecks.cpp

  DEPENDS
  ${tablegen_deps}

  LINK_LIBS
    lldbCore
    lldbExpression
    lldbHost
    lldbInterpreter
    lldbSymbol
    lldbTarget
    lldbUtility
    lldbPluginCPlusPlusLanguage
    lldbPluginCPPRuntime
  CLANG_LIBS
    clangAST
    clangCodeGen
    clangDriver
    clangEdit
    clangFrontend
    clangLex
    clangParse
    clangRewrite
    clangRewriteFrontend
    clangSema
    clangSerialization
  LINK_COMPONENTS
    Core
    ExecutionEngine
    ipo
    MCJIT
    Support
  )
