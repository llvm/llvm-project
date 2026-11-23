# Solaris code uses kstat, so specify dependency explicitly for shared builds.
if ("${CMAKE_SYSTEM_NAME}" MATCHES "SunOS")
  set(system_libs kstat)
endif()

add_llvm_component_library(LLVMTargetParser
  AArch64TargetParser.cpp
  AMDGPUTargetParser.cpp
  ARMTargetParserCommon.cpp
  ARMTargetParser.cpp
  AVRTargetParser.cpp
  CSKYTargetParser.cpp
  Host.cpp
  IntelGPUTargetParser.cpp
  LoongArchTargetParser.cpp
  NVPTXTargetParser.cpp
  PPCTargetParser.cpp
  RISCVISAInfo.cpp
  RISCVTargetParser.cpp
  SubtargetFeature.cpp
  TargetParser.cpp
  Triple.cpp
  TargetDataLayout.cpp
  X86TargetParser.cpp
  XtensaTargetParser.cpp

  ADDITIONAL_HEADER_DIRS
  Unix
  Windows

  LINK_LIBS
  ${system_libs}

  LINK_COMPONENTS
  Support

  DEPENDS
  target_parser_gen
  )
