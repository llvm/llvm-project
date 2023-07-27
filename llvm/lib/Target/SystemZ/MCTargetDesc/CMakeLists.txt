add_llvm_component_library(LLVMSystemZDesc
  SystemZInstPrinter.cpp
  SystemZMCAsmBackend.cpp
  SystemZMCAsmInfo.cpp
  SystemZMCCodeEmitter.cpp
  SystemZMCExpr.cpp
  SystemZMCObjectWriter.cpp
  SystemZMCTargetDesc.cpp

  LINK_COMPONENTS
  CodeGenTypes
  MC
  Support
  SystemZInfo

  ADD_TO_COMPONENT
  SystemZ
  )
