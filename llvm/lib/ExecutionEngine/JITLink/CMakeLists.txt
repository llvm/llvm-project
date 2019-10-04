add_llvm_library(LLVMJITLink
  JITLink.cpp
  JITLinkGeneric.cpp
  JITLinkMemoryManager.cpp
  EHFrameSupport.cpp
  MachO.cpp
  MachO_x86_64.cpp
  MachOLinkGraphBuilder.cpp

  DEPENDS
  intrinsics_gen
  )
