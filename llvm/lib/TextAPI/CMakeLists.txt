add_llvm_library(LLVMTextAPI
  ELF/ELFStub.cpp
  ELF/TBEHandler.cpp

  ADDITIONAL_HEADER_DIRS
  "${LLVM_MAIN_INCLUDE_DIR}/llvm/TextAPI"
)
