add_llvm_component_library(LLVMHelloNew
  HelloWorld.cpp

  DEPENDS
  intrinsics_gen
  )
