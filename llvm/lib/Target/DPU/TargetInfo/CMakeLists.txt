add_llvm_component_library(LLVMDPUInfo
  DPUTargetInfo.cpp

  LINK_COMPONENTS
  Support

  ADD_TO_COMPONENT
  DPU
  )
