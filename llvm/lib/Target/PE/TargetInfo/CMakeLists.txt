add_llvm_component_library(LLVMPEInfo
  PETargetInfo.cpp
  LINK_COMPONENTS #指定依赖的组件Machine Code
  MC
  ADD_TO_COMPONENT #归属于父组件PE
  PE
)