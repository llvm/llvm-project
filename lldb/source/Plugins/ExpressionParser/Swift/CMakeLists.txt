add_lldb_library(lldbPluginExpressionParserSwift PLUGIN
  SwiftASTManipulator.cpp
  SwiftExpressionParser.cpp
  SwiftExpressionVariable.cpp
  SwiftPersistentExpressionState.cpp
  SwiftREPL.cpp
  SwiftREPLMaterializer.cpp
  SwiftUserExpression.cpp
  SwiftSILManipulator.cpp

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
