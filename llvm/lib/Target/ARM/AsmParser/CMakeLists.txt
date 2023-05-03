add_llvm_component_library(LLVMARMAsmParser
  ARMAsmParser.cpp

  LINK_COMPONENTS
  ARMDesc
  ARMInfo
  ARMUtils
  CodeGen
  MC
  MCParser
  Support
  TargetParser

  ADD_TO_COMPONENT
  ARM
  )
