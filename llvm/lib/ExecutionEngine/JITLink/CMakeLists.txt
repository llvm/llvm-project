add_llvm_component_library(LLVMJITLink
  JITLink.cpp
  JITLinkGeneric.cpp
  JITLinkMemoryManager.cpp
  EHFrameSupport.cpp

  # Formats:

  # MachO
  MachO.cpp
  MachO_arm64.cpp
  MachO_x86_64.cpp
  MachOLinkGraphBuilder.cpp

  # ELF

  ELF.cpp
  ELF_x86_64.cpp

  # Architectures:
  x86_64.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/ExecutionEngine/JITLink

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  BinaryFormat
  Object
  OrcTargetProcess
  Support
  )
 
target_link_libraries(LLVMJITLink
  PRIVATE
  LLVMObject
  LLVMOrcTargetProcess
  LLVMSupport
)
