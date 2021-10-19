add_llvm_component_library(LLVMXCoreDisassembler
  XCoreDisassembler.cpp

  LINK_COMPONENTS
  MCDisassembler
  Support
  XCoreInfo

  ADD_TO_COMPONENT
  XCore
  )
