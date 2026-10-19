add_llvm_component_library(LLVMToy64Info
  Toy64TargetInfo.cpp

  LINK_COMPONENTS
  MC
  Support

  ADD_TO_COMPONENT
  Toy64
)