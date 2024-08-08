add_llvm_component_library(LLVMFrontendOffloadingAMDGPU
  ObjectUtilities.cpp

  LINK_COMPONENTS
  Support
  BinaryFormat
  Object
)
