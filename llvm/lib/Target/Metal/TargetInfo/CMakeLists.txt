add_llvm_component_library(LLVMMetalInfo
  MetalTargetInfo.cpp

  LINK_COMPONENTS
  MC
  Support
  TargetParser

  ADD_TO_COMPONENT
  Metal
  )

add_dependencies(LLVMMetalInfo MetalCommonTableGen)
