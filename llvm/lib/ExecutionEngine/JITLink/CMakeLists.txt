# COFFOptions tablegen is needed unless this is an AArch64-only EJIT build.
if(NOT (EJIT_TRIM_LLVM_BACKEND AND LLVM_TARGETS_TO_BUILD STREQUAL "AArch64"))
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

# ELF backends for non-AArch64/non-x86 arches — excluded in AArch64 EJIT builds,
# included in normal and X86 EJIT builds.
set(JITLINK_ELF_OTHER
  ELF_aarch32.cpp  ELF_loongarch.cpp  ELF_ppc64.cpp
  ELF_riscv.cpp
  )

# XCOFF format and non-AArch64/non-x86 arch helpers — excluded in AArch64 EJIT
# builds, included in normal and X86 EJIT builds.
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

if(EJIT_TRIM_LLVM_BACKEND AND LLVM_TARGETS_TO_BUILD STREQUAL "AArch64")
  # AArch64 with EJIT_TRIM_LLVM_BACKEND: ELF only, AArch64 only.  aarch32.cpp
  # needed by LinkGraphLinkingLayer.cpp (hasTargetFlags).
  list(APPEND JITLINK_SOURCES aarch32.cpp)
else()
  # Normal build, X86 EJIT, or EXPERIMENTAL: include all format/arch backends.
  list(APPEND JITLINK_SOURCES ${JITLINK_MACHO_COFF})
  list(APPEND JITLINK_SOURCES ${JITLINK_ELF_OTHER})
  list(APPEND JITLINK_SOURCES ${JITLINK_BLOAT})
  list(APPEND JITLINK_SOURCES ${JITLINK_ARCH_X86})
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
