add_llvm_tool(opt
  opt.cpp
  DEPENDS
  intrinsics_gen
  SUPPORT_PLUGINS
  )
target_link_libraries(opt PRIVATE LLVMOptDriver)
export_executable_symbols_for_plugins(opt)
