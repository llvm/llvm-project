add_llvm_component_library(LLVMToy64Desc
  Toy64MCTargetDesc.cpp

  LINK_COMPONENTS
  MC
  Support

  ADD_TO_COMPONENT
  Toy64
)