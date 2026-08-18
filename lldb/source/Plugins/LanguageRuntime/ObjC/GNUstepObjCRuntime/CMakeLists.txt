add_lldb_library(lldbPluginGNUstepObjCRuntime PLUGIN
  GNUstepObjCClassDescriptor.cpp
  GNUstepObjCDeclVendor.cpp
  GNUstepObjCTypeEncodingParser.cpp
  GNUstepObjCRuntime.cpp
  GNUstepThreadPlanStepThroughObjCTrampoline.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbExpression
    lldbHost
    lldbInterpreter
    lldbSymbol
    lldbTarget
    lldbUtility
    lldbValueObject
    lldbPluginExpressionParserClang
    lldbPluginTypeSystemClang
  CLANG_LIBS
    clangAST
  )
