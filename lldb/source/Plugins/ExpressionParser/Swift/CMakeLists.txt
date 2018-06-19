if(NOT LLDB_BUILT_STANDALONE)
  set(tablegen_deps intrinsics_gen)
endif()

add_lldb_library(lldbPluginExpressionParserSwift PLUGIN
  SwiftASTManipulator.cpp
  SwiftExpressionParser.cpp
  SwiftExpressionVariable.cpp
  SwiftPersistentExpressionState.cpp
  SwiftREPL.cpp
  SwiftREPLMaterializer.cpp
  SwiftUserExpression.cpp
  SwiftSILManipulator.cpp

  DEPENDS
  ${tablegen_deps}

  LINK_LIBS
    lldbCore
    lldbDataFormatters
    lldbExpression
    lldbHost
    lldbInterpreter
    lldbSymbol
    lldbTarget
    lldbUtility
    swiftAST
    swiftBasic
    swiftClangImporter
    swiftFrontend
    swiftIDE
    swiftParse
    swiftSIL
    swiftSILOptimizer
    swiftSerialization
    clangAST
    clangBasic
    clangRewrite

  LINK_COMPONENTS
    Support
    Core
  )
