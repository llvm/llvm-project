add_llvm_component_library(LLVMAArch64TargetMCA
  AArch64CustomBehaviour.cpp

  LINK_COMPONENTS
  AArch64Desc
  AArch64Info
  MCA
  Support

  ADD_TO_COMPONENT
  AArch64
  )
