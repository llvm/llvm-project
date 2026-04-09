# TargetInfo
add_llvm_component_library(LLVMLX32Info
  LX32TargetInfo.cpp

  LINK_COMPONENTS
  MC
  Support

  ADD_TO_COMPONENT
  LX32
)