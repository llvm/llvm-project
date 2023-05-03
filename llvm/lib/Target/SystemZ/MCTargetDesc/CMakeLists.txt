add_llvm_component_library(LLVMSystemZDesc
  SystemZInstPrinter.cpp
  SystemZMCAsmBackend.cpp
  SystemZMCAsmInfo.cpp
  SystemZMCCodeEmitter.cpp
  SystemZMCObjectWriter.cpp
  SystemZMCTargetDesc.cpp

  LINK_COMPONENTS
  CodeGen
  MC
  Support
  SystemZInfo

  ADD_TO_COMPONENT
  SystemZ
  )
