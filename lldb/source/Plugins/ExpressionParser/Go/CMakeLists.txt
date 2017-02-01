add_lldb_library(lldbPluginExpressionParserGo PLUGIN
	GoLexer.cpp
	GoParser.cpp
	GoUserExpression.cpp

  LINK_LIBS
    lldbCore
    lldbExpression
    lldbSymbol
    lldbTarget
  LINK_COMPONENTS
    Support
  )
