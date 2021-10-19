add_llvm_component_library(LLVMWebAssemblyUtils
  WebAssemblyUtilities.cpp
  WebAssemblyTypeUtilities.cpp

  LINK_COMPONENTS
  CodeGen
  Core
  MC
  Support

  ADD_TO_COMPONENT
  WebAssembly
  )
