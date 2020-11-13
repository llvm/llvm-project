add_llvm_component_library(LLVMInterfaceStub
  ELFObjHandler.cpp
  ELFStub.cpp
  TBEHandler.cpp

  LINK_COMPONENTS
  Object
  Support
)
