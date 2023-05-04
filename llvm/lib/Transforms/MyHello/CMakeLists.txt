add_llvm_library( LLVMMyHello MODULE
  Hello.cpp

  PLUGIN_TOOL
  opt
  )