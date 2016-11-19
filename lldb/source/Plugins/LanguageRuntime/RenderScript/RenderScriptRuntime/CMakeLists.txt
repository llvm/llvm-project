if(NOT LLDB_BUILT_STANDALONE)
  set(tablegen_deps intrinsics_gen)
endif()


add_lldb_library(lldbPluginRenderScriptRuntime
  RenderScriptRuntime.cpp
  RenderScriptExpressionOpts.cpp
  RenderScriptx86ABIFixups.cpp
  RenderScriptScriptGroup.cpp

  DEPENDS
  ${tablegen_deps}
  )
