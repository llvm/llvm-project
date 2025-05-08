# Modified by Sunscreen under the AGPLv3 license; see the README at the
# repository root for more information

add_llvm_component_library(LLVMParasolInfo
  ParasolTargetInfo.cpp

  ADD_TO_COMPONENT
  Parasol
)
