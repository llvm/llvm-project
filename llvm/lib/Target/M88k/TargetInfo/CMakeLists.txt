add_llvm_component_library(LLVMM88kInfo
  M88kTargetInfo.cpp

  LINK_COMPONENTS
  Support

  ADD_TO_COMPONENT
  M88k
  )
