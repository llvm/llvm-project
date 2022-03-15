add_llvm_component_library(LLVMJITLink
  DWARFRecordSectionSplitter.cpp
  EHFrameSupport.cpp
  JITLink.cpp
  JITLinkGeneric.cpp
  JITLinkMemoryManager.cpp
  MemoryFlags.cpp

  # Formats:

  # MachO
  MachO.cpp
  MachO_arm64.cpp
  MachO_x86_64.cpp
  MachOLinkGraphBuilder.cpp

  # ELF
  ELF.cpp
  ELFLinkGraphBuilder.cpp
  ELF_aarch64.cpp
  ELF_riscv.cpp
  ELF_x86_64.cpp

  # Architectures:
  aarch64.cpp
  riscv.cpp
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
  LLVMOrcShared
  LLVMOrcTargetProcess
  LLVMSupport
)
