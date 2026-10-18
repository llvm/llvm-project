add_llvm_component_library(LLVMAGXAsmParser
  AGXAsmParser.cpp

  LINK_COMPONENTS
  AGXDesc
  AGXInfo
  AGXUtils
  MC
  MCParser
  Support
  TargetParser

  ADD_TO_COMPONENT
  AGX
  )
