set(LLVM_TARGET_DEFINITIONS COFFOptions.td)
tablegen(LLVM COFFOptions.inc -gen-opt-parser-defs)
add_public_tablegen_target(JITLinkTableGen)

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

  # COFF
  COFF.cpp
  COFFDirectiveParser.cpp
  COFFLinkGraphBuilder.cpp
  COFF_x86_64.cpp

  # Architectures:
  aarch64.cpp
  riscv.cpp
  x86_64.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/ExecutionEngine/JITLink

  DEPENDS
  intrinsics_gen
  JITLinkTableGen

  LINK_COMPONENTS
  BinaryFormat
  Object
  Option
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
