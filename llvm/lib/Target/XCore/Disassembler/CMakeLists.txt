add_llvm_component_library(LLVMXCoreDisassembler
  XCoreDisassembler.cpp

  LINK_COMPONENTS
  CodeGenTypes
  MC
  MCDisassembler
  Support
  XCoreInfo
  XCoreDesc

  ADD_TO_COMPONENT
  XCore
  )
