add_llvm_component_library(LLVMAIRInfo
  AIRTargetInfo.cpp

  LINK_COMPONENTS
  MC
  Support
  TargetParser

  ADD_TO_COMPONENT
  AIR
  )

add_dependencies(LLVMAIRInfo AIRCommonTableGen)
