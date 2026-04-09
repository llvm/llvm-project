add_llvm_component_library(LLVMLX32AsmParser
  LX32AsmParser.cpp

  LINK_COMPONENTS
  MC
  MCParser
  Support
  LX32Desc
  LX32Info

  ADD_TO_COMPONENT
  LX32
  )
