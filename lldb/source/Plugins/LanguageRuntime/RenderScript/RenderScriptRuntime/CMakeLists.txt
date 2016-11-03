add_lldb_library(lldbPluginRenderScriptRuntime
  RenderScriptRuntime.cpp
  RenderScriptExpressionOpts.cpp
  RenderScriptx86ABIFixups.cpp
  RenderScriptScriptGroup.cpp
  )

if(NOT LLDB_BUILT_STANDALONE)
  add_dependencies(lldbPluginRenderScriptRuntime intrinsics_gen)
endif()
