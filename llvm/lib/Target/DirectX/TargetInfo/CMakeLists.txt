add_llvm_component_library(LLVMDirectXInfo
  DirectXTargetInfo.cpp

  LINK_COMPONENTS
  MC
  Support

  ADD_TO_COMPONENT
  DirectX
)

add_dependencies(LLVMDirectXInfo DirectXCommonTableGen)
