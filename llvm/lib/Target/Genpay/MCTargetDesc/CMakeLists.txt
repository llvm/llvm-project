add_llvm_component_library(LLVMGenpayDesc
  GenpayInstPrinter.cpp
  GenpayMCInstLower.cpp
  GenpayMCTargetDesc.cpp

  LINK_COMPONENTS
  MC
  GenpayInfo
  Support
  TargetParser

  ADD_TO_COMPONENT
  Genpay
  )

add_dependencies(LLVMGenpayDesc GenpayCommonTableGen)
