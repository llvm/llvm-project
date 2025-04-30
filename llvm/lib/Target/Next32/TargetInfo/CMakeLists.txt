add_llvm_component_library(LLVMNext32Info
  Next32TargetInfo.cpp
  Next32BaseInfo.cpp

  LINK_COMPONENTS
  Support
  CodeGen

  ADD_TO_COMPONENT
  Next32
  )
