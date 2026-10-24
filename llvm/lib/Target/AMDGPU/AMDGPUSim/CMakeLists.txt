add_llvm_component_library(LLVMAMDGPUSim
  HWModel.cpp
  Simulator.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_COMPONENTS
  Support

  ADD_TO_COMPONENT
  AMDGPU
  )
