add_llvm_component_library(LLVMDirectXDesc
  DirectXMCTargetDesc.cpp

  LINK_COMPONENTS
  DirectXInfo
  Support

  ADD_TO_COMPONENT
  DirectX
  )
