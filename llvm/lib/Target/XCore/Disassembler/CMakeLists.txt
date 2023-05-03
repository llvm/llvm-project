add_llvm_component_library(LLVMXCoreDisassembler
  XCoreDisassembler.cpp

  LINK_COMPONENTS
  CodeGen
  MC
  MCDisassembler
  Support
  XCoreInfo

  ADD_TO_COMPONENT
  XCore
  )
