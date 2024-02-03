add_llvm_tool(opt-printplugin
  opt-printplugin.cpp
  DEPENDS
  intrinsics_gen
  SUPPORT_PLUGINS
  )
target_link_libraries(opt-printplugin PRIVATE LLVMOptDriver)
export_executable_symbols_for_plugins(opt-printplugin)
