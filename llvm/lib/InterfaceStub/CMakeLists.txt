add_llvm_component_library(LLVMInterfaceStub
  ELFObjHandler.cpp
  ELFStub.cpp
  TBEHandler.cpp

  ADDITIONAL_HEADER_DIRS
  "${LLVM_MAIN_INCLUDE_DIR}/llvm/InterfaceStub"
)
