if(NOT LLDB_BUILT_STANDALONE)
  set(tablegen_deps intrinsics_gen)
endif()


add_lldb_library(lldbPluginRenderScriptRuntime PLUGIN
  RenderScriptRuntime.cpp
  RenderScriptExpressionOpts.cpp
  RenderScriptx86ABIFixups.cpp
  RenderScriptScriptGroup.cpp

  DEPENDS
  ${tablegen_deps}

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
