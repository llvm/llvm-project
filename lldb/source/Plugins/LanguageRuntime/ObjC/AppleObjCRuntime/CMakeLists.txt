lldb_tablegen(AppleObjCRuntimeOptions.inc -gen-lldb-option-defs
  SOURCE AppleObjCRuntimeOptions.td
  TARGET LLDBPluginAppleObjCRuntimeOptionsGen)

add_lldb_library(lldbPluginAppleObjCRuntime PLUGIN
  AppleObjCRuntime.cpp
  AppleObjCRuntimeV2.cpp
  AppleObjCTrampolineHandler.cpp
  AppleObjCDeclVendor.cpp
  AppleThreadPlanStepThroughObjCTrampoline.cpp
  AppleObjCClassDescriptorV2.cpp
  AppleObjCTypeEncodingParser.cpp

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
    lldbPluginCPPRuntime
    lldbPluginTypeSystemClang
  CLANG_LIBS
    clangAST
  )

add_dependencies(lldbPluginAppleObjCRuntime
  LLDBPluginAppleObjCRuntimeOptionsGen)
