add_llvm_component_library(LLVMSuperHAsmParser
  SuperHAsmParser.cpp

  LINK_COMPONENTS
  MC
  MCParser
  SuperHDesc
  SuperHInfo
  Support
  TargetParser

  ADD_TO_COMPONENT
  SuperH
  )
