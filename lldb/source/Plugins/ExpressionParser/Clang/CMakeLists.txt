if(NOT LLDB_BUILT_STANDALONE)
  set(tablegen_deps intrinsics_gen)
endif()

add_lldb_library(lldbPluginExpressionParserClang PLUGIN
  ASTDumper.cpp
  ASTResultSynthesizer.cpp
  ASTStructExtractor.cpp
  ClangASTSource.cpp
  ClangExpressionDeclMap.cpp
  ClangExpressionParser.cpp
  ClangExpressionVariable.cpp
  ClangFunctionCaller.cpp
  ClangModulesDeclVendor.cpp
  ClangPersistentVariables.cpp
  ClangUserExpression.cpp
  ClangUtilityFunction.cpp
  IRForTarget.cpp

  DEPENDS
  ${tablegen_deps}

  LINK_LIBS
    clangAST
    clangCodeGen
    clangEdit
    clangFrontend
    clangLex
    clangParse
    clangRewrite
    clangRewriteFrontend
    clangSema
    clangSerialization
    lldbCore
    lldbExpression
    lldbHost
    lldbInterpreter
    lldbSymbol
    lldbTarget
    lldbUtility
    lldbPluginCPlusPlusLanguage
  LINK_COMPONENTS
    Core
    ExecutionEngine
    ipo
    MCJIT
    Support
  )
