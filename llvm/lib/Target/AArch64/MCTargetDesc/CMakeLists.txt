set(AARCH64DESC_SOURCES
  AArch64AsmBackend.cpp
  AArch64ELFObjectWriter.cpp
  AArch64ELFStreamer.cpp
  AArch64InstPrinter.cpp
  AArch64MCAsmInfo.cpp
  AArch64MCCodeEmitter.cpp
  AArch64MCExpr.cpp
  AArch64MCTargetDesc.cpp
  AArch64TargetStreamer.cpp
  )

set(AARCH64DESC_OPT_SOURCES
  AArch64MachObjectWriter.cpp
  AArch64WinCOFFObjectWriter.cpp
  AArch64WinCOFFStreamer.cpp
  )

if(NOT EJIT_TRIM_LLVM_BACKEND_EXPERIMENTAL)
  list(APPEND AARCH64DESC_SOURCES ${AARCH64DESC_OPT_SOURCES})
endif()

add_llvm_component_library(LLVMAArch64Desc
  ${AARCH64DESC_SOURCES}

  PARTIAL_SOURCES_INTENDED

  LINK_COMPONENTS
  AArch64Info
  AArch64Utils
  BinaryFormat
  CodeGenTypes
  MC
  Support
  TargetParser

  ADD_TO_COMPONENT
  AArch64
  )

add_dependencies(LLVMAArch64Desc AArch64CommonTableGen)

# Hack: we need to include 'main' target directory to grab private headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_BINARY_DIR}/..)
