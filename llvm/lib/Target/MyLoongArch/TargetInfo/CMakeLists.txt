add_llvm_component_library(LLVMMyLoongArchInfo
  MyLoongArchTargetInfo.cpp

  LINK_COMPONENTS
  MC
  
  ADD_TO_COMPONENT
  MyLoongArch
)