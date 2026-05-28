if(NOT EJIT_BARE_METAL)
  set(LLVM_TARGET_DEFINITIONS COFFOptions.td)
  tablegen(LLVM COFFOptions.inc -gen-opt-parser-defs)
  add_public_tablegen_target(JITLinkTableGen)
endif()

set(JITLINK_SOURCES
  DWARFRecordSectionSplitter.cpp
  EHFrameSupport.cpp
  JITLink.cpp
  JITLinkGeneric.cpp
  JITLinkMemoryManager.cpp

  # ELF (AArch64 only for bare-metal)
  ELF.cpp
  ELFLinkGraphBuilder.cpp
  ELF_aarch64.cpp

  # Architectures (AArch64 only)
  aarch64.cpp
  )

set(JITLINK_OPT_SOURCES
  CompactUnwindSupport.cpp

  # MachO
  MachO.cpp
  MachO_arm64.cpp
  MachO_x86_64.cpp
  MachOLinkGraphBuilder.cpp

  # ELF (other arches)
  ELF_aarch32.cpp
  ELF_loongarch.cpp
  ELF_ppc64.cpp
  ELF_riscv.cpp
  ELF_x86.cpp
  ELF_x86_64.cpp

  # COFF
  COFF.cpp
  COFFDirectiveParser.cpp
  COFFLinkGraphBuilder.cpp
  COFF_x86_64.cpp

  # XCOFF
  XCOFF.cpp
  XCOFF_ppc64.cpp
  XCOFFLinkGraphBuilder.cpp

  # Architectures (other)
  aarch32.cpp
  loongarch.cpp
  ppc64.cpp
  riscv.cpp
  x86.cpp
  x86_64.cpp
  )

if(NOT EJIT_BARE_METAL)
  list(APPEND JITLINK_SOURCES ${JITLINK_OPT_SOURCES})
endif()

set(JITLINK_DEPENDS intrinsics_gen)
if(NOT EJIT_BARE_METAL)
  list(APPEND JITLINK_DEPENDS JITLinkTableGen)
endif()

add_llvm_component_library(LLVMJITLink
  ${JITLINK_SOURCES}

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/ExecutionEngine/JITLink

  DEPENDS
  ${JITLINK_DEPENDS}

  LINK_COMPONENTS
  BinaryFormat
  Object
  Option
  OrcTargetProcess
  Support
  TargetParser
  )

target_link_libraries(LLVMJITLink
  PRIVATE
  LLVMObject
  LLVMOrcShared
  LLVMOrcTargetProcess
  LLVMSupport
  LLVMTargetParser
)
