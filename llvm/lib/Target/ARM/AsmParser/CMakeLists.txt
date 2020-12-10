add_llvm_component_library(LLVMARMAsmParser
  ARMAsmParser.cpp

  LINK_COMPONENTS
  ARMDesc
  ARMInfo
  MC
  MCParser
  Support
  ARMUtils

  ADD_TO_COMPONENT
  ARM
  )
