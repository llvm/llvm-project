add_llvm_component_library(LLVMDirectXDesc
  DirectXContainerObjectWriter.cpp
  DirectXMCTargetDesc.cpp

  LINK_COMPONENTS
  DirectXInfo
  Support

  ADD_TO_COMPONENT
  DirectX
  )
