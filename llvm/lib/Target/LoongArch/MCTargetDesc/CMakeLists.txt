add_llvm_component_library(LLVMLoongArchDesc
  LoongArchMCTargetDesc.cpp

  LINK_COMPONENTS
  MC
  LoongArchInfo
  Support

  ADD_TO_COMPONENT
  LoongArch
  )
