add_llvm_library(LLVMJITLink
  JITLink.cpp
  JITLinkGeneric.cpp
  JITLink_EHFrameSupport.cpp
  JITLink_MachO.cpp
  JITLink_MachO_x86_64.cpp
  MachOAtomGraphBuilder.cpp

  DEPENDS
  intrinsics_gen
  )
