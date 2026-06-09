if(NOT EJIT_TRIM_LLVM_BACKEND_EXPERIMENTAL)
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

  # ELF (always needed)
  ELF.cpp
  ELFLinkGraphBuilder.cpp
  ELF_aarch64.cpp

  # Base architecture support (needed by format backends)
  aarch64.cpp
  )

# Format-specific backends needed by x86 (macOS/Windows)
set(JITLINK_MACHO_COFF
  CompactUnwindSupport.cpp
  MachO.cpp  MachO_arm64.cpp  MachO_x86_64.cpp  MachOLinkGraphBuilder.cpp
  COFF.cpp  COFFDirectiveParser.cpp  COFFLinkGraphBuilder.cpp  COFF_x86_64.cpp
  )

# ELF arches not needed by EJIT (keep only current target's arch)
set(JITLINK_ELF_OTHER
  ELF_aarch32.cpp  ELF_loongarch.cpp  ELF_ppc64.cpp
  ELF_riscv.cpp
  )

# Bloat — not needed by any EJIT arch
set(JITLINK_BLOAT
  XCOFF.cpp  XCOFF_ppc64.cpp  XCOFFLinkGraphBuilder.cpp
  # Arch support (non-AArch64, non-x86)
  aarch32.cpp  loongarch.cpp  ppc64.cpp  riscv.cpp
  )

set(JITLINK_ARCH_X86
  ELF_x86.cpp  ELF_x86_64.cpp
  x86.cpp  x86_64.cpp
  )

set(JITLINK_DEPENDS intrinsics_gen)

if(NOT EJIT_TRIM_LLVM_BACKEND_EXPERIMENTAL)
  # Normal build: include everything
  list(APPEND JITLINK_SOURCES ${JITLINK_MACHO_COFF})
  list(APPEND JITLINK_SOURCES ${JITLINK_ELF_OTHER})
  list(APPEND JITLINK_SOURCES ${JITLINK_BLOAT})
  list(APPEND JITLINK_SOURCES ${JITLINK_ARCH_X86})
  list(APPEND JITLINK_DEPENDS JITLinkTableGen)
elseif(LLVM_TARGETS_TO_BUILD STREQUAL "AArch64")
  # AArch64 bare-metal: ELF only, AArch64 only.  Include aarch32 to
  # satisfy extern declarations transitively included via ELF_aarch32.h.
  list(APPEND JITLINK_SOURCES aarch32.cpp)
else()
  # x86 bare-metal: ELF + MachO + COFF, x86 arch support
  list(APPEND JITLINK_SOURCES ${JITLINK_MACHO_COFF})
  list(APPEND JITLINK_SOURCES ${JITLINK_ARCH_X86})
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
