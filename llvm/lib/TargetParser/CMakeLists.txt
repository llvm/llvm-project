
add_llvm_component_library(LLVMTargetParser
  AArch64TargetParser.cpp
  ARMTargetParserCommon.cpp
  ARMTargetParser.cpp
  CSKYTargetParser.cpp
  Host.cpp
  LoongArchTargetParser.cpp
  TargetParser.cpp
  Triple.cpp
  X86TargetParser.cpp

  ADDITIONAL_HEADER_DIRS
  Unix
  Windows

  LINK_COMPONENTS
  Support
  )
