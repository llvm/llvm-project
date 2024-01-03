add_llvm_component_library(LLVMTextAPIBinaryReader
  DylibReader.cpp

  LINK_COMPONENTS
  Support
  Object
  TextAPI
  TargetParser
)
