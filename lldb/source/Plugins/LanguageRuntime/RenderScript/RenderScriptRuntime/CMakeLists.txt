add_lldb_library(lldbPluginRenderScriptRuntime PLUGIN
  RenderScriptRuntime.cpp
  RenderScriptExpressionOpts.cpp
  RenderScriptx86ABIFixups.cpp
  RenderScriptScriptGroup.cpp

  DEPENDS
  intrinsics_gen

  LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbDataFormatters
    lldbExpression
    lldbHost
    lldbInterpreter
    lldbSymbol
    lldbTarget
  CLANG_LIBS
    clangBasic
  LINK_COMPONENTS
    Core
    IRReader
    Support
    Target
  )
