add_llvm_component_library(LLVMSC32Info
  SC32TargetInfo.cpp

  LINK_COMPONENTS
  Support
  MC

  ADD_TO_COMPONENT
  SC32
  )
