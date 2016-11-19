if(NOT LLDB_BUILT_STANDALONE)
  set(tablegen_deps intrinsics_gen)
endif()

add_lldb_library(lldbPluginExpressionParserClang
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
  )
