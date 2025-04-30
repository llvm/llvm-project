add_llvm_component_library(LLVMNext32AsmParser
  Next32AsmParser.cpp

  LINK_COMPONENTS
  MC
  MCParser
  Support
  Next32Desc
  Next32Info

  ADD_TO_COMPONENT
  Next32
  )
